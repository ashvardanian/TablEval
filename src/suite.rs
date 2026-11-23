//! The four query suites and the dataset folders they run over.
//!
//! A dataset folder holds `<table>.part_NNN.parquet` files, read in natural order. Its tables tell the
//! suite apart: `trips` is NYC Taxi, `lineitem` with the other seven TPC-H tables is TPC-H, and `hits` with
//! the `queries.sql` that `tabl-download` fetched is ClickBench.
//!
//! The documents suite runs over JSON documents already stored, partitioned by key, one store per partition. Its
//! dataset folder holds the key draws `tabl-draw-keys` wrote, `<count>.arrow` each, which every backend reads, so
//! every system gathers the same keys. Each partition is one table, `<collection>_p<partition>`.

use std::{
    collections::HashMap,
    fs::File,
    path::{Path, PathBuf},
    sync::Arc,
};

use arrow::{
    array::{Array, ArrayRef, AsArray, FixedSizeBinaryArray, RecordBatch, UInt8Array},
    datatypes::{DataType, Field, Schema, SchemaRef, UInt8Type},
    ipc::{reader::FileReader as IpcFileReader, writer::FileWriter as IpcFileWriter},
};
use parquet::file::reader::{FileReader, SerializedFileReader};

use crate::{answer::Answer, backend::Table, parse_count, spell_count, Result};

#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Suite {
    Taxi,
    Tpch,
    Clickbench,
    Documents,
}

impl Suite {
    /// The table `--rows` cuts, and the one throughput is counted in.
    pub fn fact_table(self) -> &'static str {
        match self {
            Self::Taxi => "trips",
            Self::Tpch => "lineitem",
            Self::Clickbench => "hits",
            Self::Documents => "documents",
        }
    }
}

/// One named query; `ties` marks a `LIMIT` over a non-unique order, where only the row count is verified.
#[derive(Clone, Debug)]
pub struct Query {
    pub name: String,
    pub sql: String,
    pub ties: bool,
    /// What a query of the documents suite runs; `None` for the SQL suites.
    pub documents: Option<Documents>,
}

/// The type a documents field converts into, in every system.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FieldType {
    Text,
    Integer,
    Real,
    Boolean,
}

impl FieldType {
    /// The Arrow type every backend lands the field's cells in.
    pub fn arrow_type(self) -> DataType {
        match self {
            Self::Text => DataType::Utf8,
            Self::Integer => DataType::Int64,
            Self::Real => DataType::Float64,
            Self::Boolean => DataType::Boolean,
        }
    }
}

/// The Arrow schema of `fields`, every column nullable, since a document may lack any field.
pub fn fields_schema(fields: &[DocumentField]) -> SchemaRef {
    let fields: Vec<Field> = fields
        .iter()
        .map(|field| Field::new(field.name, field.field_type.arrow_type(), true))
        .collect();
    Arc::new(Schema::new(fields))
}

/// `cells` behind a `key` column of the 16-byte keys laid end to end in `keys`, as every documents query answers.
pub fn keyed_batch(keys: Vec<u8>, cells: &RecordBatch) -> Result<RecordBatch> {
    let keys = FixedSizeBinaryArray::try_new(16, keys.into(), None).map_err(|e| e.to_string())?;
    let mut fields = vec![Arc::new(Field::new("key", DataType::FixedSizeBinary(16), false))];
    fields.extend(cells.schema().fields().iter().cloned());
    let mut columns: Vec<ArrayRef> = vec![Arc::new(keys)];
    columns.extend(cells.columns().iter().cloned());
    RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).map_err(|e| e.to_string())
}

/// One scalar field of a documents collection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DocumentField {
    pub name: &'static str,
    pub field_type: FieldType,
}

const fn field(name: &'static str, field_type: FieldType) -> DocumentField {
    DocumentField { name, field_type }
}

/// AlphaFold's scalar fields; the lists `go_terms`, `ec_numbers` and `pfam_domains` would gather as collisions.
const ALPHAFOLD: [DocumentField; 10] = [
    field("uniprot_accession", FieldType::Text),
    field("taxonomy_id", FieldType::Integer),
    field("organism", FieldType::Text),
    field("sequence", FieldType::Text),
    field("afdb_entry", FieldType::Text),
    field("fragment", FieldType::Integer),
    field("domain", FieldType::Text),
    field("description", FieldType::Text),
    field("reviewed", FieldType::Boolean),
    field("gene", FieldType::Text),
];

const OAS_UNPAIRED: [DocumentField; 25] = [
    field("v_call", FieldType::Text),
    field("redundancy", FieldType::Integer),
    field("v_identity", FieldType::Real),
    field("sequence", FieldType::Text),
    field("cdr3", FieldType::Text),
    field("regions", FieldType::Text),
    field("d_call", FieldType::Text),
    field("j_call", FieldType::Text),
    field("locus", FieldType::Text),
    field("j_identity", FieldType::Real),
    field("study", FieldType::Text),
    field("file", FieldType::Text),
    field("run", FieldType::Text),
    field("author", FieldType::Text),
    field("link", FieldType::Text),
    field("species", FieldType::Text),
    field("age", FieldType::Text),
    field("b_source", FieldType::Text),
    field("b_type", FieldType::Text),
    field("vaccine", FieldType::Text),
    field("disease", FieldType::Text),
    field("subject", FieldType::Text),
    field("longitudinal", FieldType::Text),
    field("isotype", FieldType::Text),
    field("chain", FieldType::Text),
];

const MOLECULES: [DocumentField; 1] = [field("smiles", FieldType::Text)];

/// The scalar fields of a USearchMolecules collection, in the order `f1` and `f3` take their prefixes.
pub fn scalar_fields(collection: &str) -> Option<&'static [DocumentField]> {
    match collection {
        "alphafold" => Some(&ALPHAFOLD),
        "oas_unpaired" => Some(&OAS_UNPAIRED),
        "pubchem" | "gdb13" | "real" => Some(&MOLECULES),
        _ => None,
    }
}

/// What a documents query runs over every partition at once.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Workload {
    /// The keys of the draw, gathered into Arrow.
    KeyedGather,
    /// As many keys per partition as the draw holds there, drawn by the system itself, then gathered.
    SampleGather,
    /// Every document.
    Scan,
}

/// A documents query: its workload, the draw it reads, and the fields that land in Arrow after each key.
#[derive(Clone, Debug)]
pub struct Documents {
    pub workload: Workload,
    /// The keys a keyed gather reads and the counts a sample draws; a scan reads the largest draw, whose keys a
    /// backend may split its walk at.
    pub keys: PathBuf,
    pub fields: &'static [DocumentField],
}

/// `alphafold_p3` → (`alphafold`, 3), the collection and partition a documents table names.
pub fn partition_of(table: &str) -> Option<(&str, usize)> {
    let (collection, partition) = table.rsplit_once("_p")?;
    Some((collection, partition.parse().ok()?))
}

/// One draw of keys: its collection, each partition's keys in drawn order, and each partition's documents.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Draw {
    pub collection: String,
    pub keys: Vec<Vec<[u8; 16]>>,
    pub rows: Vec<u64>,
}

/// The seed every draw starts from, so a draw repeated over the same inputs lands on the same keys.
const DRAW_SEED: u64 = 0x7461_626c_6576_616c;

/// SplitMix64, the generator behind every draw.
struct Generator(u64);

impl Generator {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    /// A uniform integer below `bound`, by the multiply-shift reduction.
    fn below(&mut self, bound: u64) -> u64 {
        ((u128::from(self.next()) * u128::from(bound)) >> 64) as u64
    }
}

fn draw_schema(collection: &str, rows: &[u64]) -> Schema {
    let rows: Vec<String> = rows.iter().map(u64::to_string).collect();
    Schema::new(vec![
        Field::new("key", DataType::FixedSizeBinary(16), false),
        Field::new("partition", DataType::UInt8, false),
    ])
    .with_metadata(HashMap::from([
        ("collection".to_owned(), collection.to_owned()),
        ("partition_rows".to_owned(), rows.join(",")),
        ("seed".to_owned(), format!("{DRAW_SEED:#x}")),
    ]))
}

impl Draw {
    /// Draws `count` keys uniformly from the partition folders `partitions`, `<collection>/p<partition>` of key-sorted
    /// Arrow files with a `key` column each, for every count at once: each partition keeps a shuffled reservoir of
    /// the largest count, and each draw splits its count over the partitions as drawing without replacement would.
    pub fn sample(partitions: &[PathBuf], counts: &[u64]) -> Result<Vec<Self>> {
        let collection = partitions
            .first()
            .and_then(|folder| folder.parent()?.file_name()?.to_str())
            .ok_or("no partition folder names its collection")?
            .to_owned();
        let largest = counts.iter().copied().max().unwrap_or(0) as usize;
        let sampled = std::thread::scope(|scope| {
            // Every partition is spawned before any is joined, so the reservoirs fill at once
            let mut threads = Vec::with_capacity(partitions.len());
            for (partition, folder) in partitions.iter().enumerate() {
                threads.push(scope.spawn(move || reservoir(folder, partition as u64, largest)));
            }
            threads
                .into_iter()
                .map(|thread| thread.join().map_err(|_| "a partition's draw panicked".to_owned())?)
                .collect::<Result<Vec<_>>>()
        })?;
        let rows: Vec<u64> = sampled.iter().map(|(_, rows)| *rows).collect();
        let mut generator = Generator(DRAW_SEED);
        let mut draws = Vec::with_capacity(counts.len());
        for &count in counts {
            let mut remaining = rows.clone();
            let mut total: u64 = remaining.iter().sum();
            let mut taken = vec![0usize; rows.len()];
            for _ in 0..count.min(total) {
                let mut rank = generator.below(total);
                let partition = remaining
                    .iter()
                    .position(|&left| match rank < left {
                        true => true,
                        false => {
                            rank -= left;
                            false
                        }
                    })
                    .ok_or("the draw ran past the documents")?;
                taken[partition] += 1;
                remaining[partition] -= 1;
                total -= 1;
            }
            let keys = sampled
                .iter()
                .zip(&taken)
                .map(|((reservoir, _), &taken)| reservoir[..taken].to_vec())
                .collect();
            draws.push(Self {
                collection: collection.clone(),
                keys,
                rows: rows.clone(),
            });
        }
        Ok(draws)
    }

    /// Reads a draw `write` saved.
    pub fn read(path: &Path) -> Result<Self> {
        let file = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let reader = IpcFileReader::try_new_buffered(file, None).map_err(|e| format!("{}: {e}", path.display()))?;
        let schema = reader.schema();
        let metadata = |name: &str| {
            schema
                .metadata()
                .get(name)
                .ok_or_else(|| format!("{} lacks the `{name}` metadata of a key draw", path.display()))
        };
        let collection = metadata("collection")?.clone();
        let rows = metadata("partition_rows")?
            .split(',')
            .map(|rows| rows.parse::<u64>().map_err(|e| format!("{}: {e}", path.display())))
            .collect::<Result<Vec<_>>>()?;
        let mut keys = vec![Vec::new(); rows.len()];
        for batch in reader {
            let batch = batch.map_err(|e| format!("{}: {e}", path.display()))?;
            let drawn = batch.column(0).as_fixed_size_binary();
            let partitions = batch.column(1).as_primitive::<UInt8Type>();
            for row in 0..batch.num_rows() {
                let partition = keys
                    .get_mut(partitions.value(row) as usize)
                    .ok_or_else(|| format!("{} names a partition past its rows", path.display()))?;
                partition.push(drawn.value(row).try_into().map_err(|_| "a key is not 16 bytes")?);
            }
        }
        Ok(Self { collection, keys, rows })
    }

    /// Saves the draw as an Arrow file of `key` and `partition` columns, its collection and partition sizes in the
    /// schema's metadata.
    pub fn write(&self, path: &Path) -> Result<()> {
        let schema = Arc::new(draw_schema(&self.collection, &self.rows));
        let keys = self.keys.iter().flatten().flatten().copied().collect::<Vec<u8>>();
        let partitions: UInt8Array = self
            .keys
            .iter()
            .enumerate()
            .flat_map(|(partition, keys)| std::iter::repeat_n(partition as u8, keys.len()))
            .collect();
        let keys = FixedSizeBinaryArray::try_new(16, keys.into(), None).map_err(|e| e.to_string())?;
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(keys), Arc::new(partitions)])
            .map_err(|e| e.to_string())?;
        let temporary = path.with_extension(format!("{}.tmp", std::process::id()));
        let file = File::create(&temporary).map_err(|e| format!("{}: {e}", temporary.display()))?;
        let mut writer = IpcFileWriter::try_new_buffered(file, &schema).map_err(|e| e.to_string())?;
        writer.write(&batch).map_err(|e| e.to_string())?;
        writer.finish().map_err(|e| e.to_string())?;
        drop(writer);
        std::fs::rename(&temporary, path).map_err(|e| format!("{}: {e}", path.display()))
    }
}

/// A shuffled uniform sample of up to `wanted` keys from the Arrow files of one partition folder, and its key count.
fn reservoir(folder: &Path, partition: u64, wanted: usize) -> Result<(Vec<[u8; 16]>, u64)> {
    let mut files = Vec::new();
    for entry in std::fs::read_dir(folder).map_err(|e| format!("{}: {e}", folder.display()))? {
        let path = entry.map_err(|e| e.to_string())?.path();
        if path.extension().is_some_and(|extension| extension == "arrow") {
            files.push(path);
        }
    }
    files.sort();
    let mut generator = Generator(DRAW_SEED ^ (partition + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15));
    let mut kept: Vec<[u8; 16]> = Vec::with_capacity(wanted);
    let mut seen = 0u64;
    for path in files {
        let file = File::open(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        let reader = IpcFileReader::try_new_buffered(file, None).map_err(|e| format!("{}: {e}", path.display()))?;
        let column = reader
            .schema()
            .index_of("key")
            .map_err(|e| format!("{}: {e}", path.display()))?;
        for batch in reader {
            let batch = batch.map_err(|e| format!("{}: {e}", path.display()))?;
            let keys = batch.column(column).as_fixed_size_binary();
            if keys.value_length() != 16 {
                return Err(format!(
                    "{} holds keys of {} bytes",
                    path.display(),
                    keys.value_length()
                ));
            }
            for row in 0..keys.len() {
                let key: [u8; 16] = keys.value(row).try_into().map_err(|_| "a key is not 16 bytes")?;
                match kept.len() < wanted {
                    true => kept.push(key),
                    false => {
                        let slot = generator.below(seen + 1) as usize;
                        if slot < wanted {
                            kept[slot] = key;
                        }
                    }
                }
                seen += 1;
            }
        }
    }
    // Fisher-Yates, so every prefix of the reservoir is a uniform sample too
    for index in (1..kept.len()).rev() {
        kept.swap(index, generator.below(index as u64 + 1) as usize);
    }
    Ok((kept, seen))
}

const TAXI: [(&str, &str); 4] = [
    ("count-by-vendor", include_str!("../sql/taxi/count-by-vendor.sql")),
    (
        "average-amount-by-passengers",
        include_str!("../sql/taxi/average-amount-by-passengers.sql"),
    ),
    (
        "count-by-passengers-and-year",
        include_str!("../sql/taxi/count-by-passengers-and-year.sql"),
    ),
    (
        "count-by-passengers-year-and-distance",
        include_str!("../sql/taxi/count-by-passengers-year-and-distance.sql"),
    ),
];

const TPCH: [&str; 22] = [
    include_str!("../sql/tpch/q01.sql"),
    include_str!("../sql/tpch/q02.sql"),
    include_str!("../sql/tpch/q03.sql"),
    include_str!("../sql/tpch/q04.sql"),
    include_str!("../sql/tpch/q05.sql"),
    include_str!("../sql/tpch/q06.sql"),
    include_str!("../sql/tpch/q07.sql"),
    include_str!("../sql/tpch/q08.sql"),
    include_str!("../sql/tpch/q09.sql"),
    include_str!("../sql/tpch/q10.sql"),
    include_str!("../sql/tpch/q11.sql"),
    include_str!("../sql/tpch/q12.sql"),
    include_str!("../sql/tpch/q13.sql"),
    include_str!("../sql/tpch/q14.sql"),
    include_str!("../sql/tpch/q15.sql"),
    include_str!("../sql/tpch/q16.sql"),
    include_str!("../sql/tpch/q17.sql"),
    include_str!("../sql/tpch/q18.sql"),
    include_str!("../sql/tpch/q19.sql"),
    include_str!("../sql/tpch/q20.sql"),
    include_str!("../sql/tpch/q21.sql"),
    include_str!("../sql/tpch/q22.sql"),
];

const TPCH_ANSWERS_SF0_01: [&str; 22] = [
    include_str!("../sql/tpch/answers-sf0.01/q01.csv"),
    include_str!("../sql/tpch/answers-sf0.01/q02.csv"),
    include_str!("../sql/tpch/answers-sf0.01/q03.csv"),
    include_str!("../sql/tpch/answers-sf0.01/q04.csv"),
    include_str!("../sql/tpch/answers-sf0.01/q05.csv"),
    include_str!("../sql/tpch/answers-sf0.01/q06.csv"),
    include_str!("../sql/tpch/answers-sf0.01/q07.csv"),
    include_str!("../sql/tpch/answers-sf0.01/q08.csv"),
    include_str!("../sql/tpch/answers-sf0.01/q09.csv"),
    include_str!("../sql/tpch/answers-sf0.01/q10.csv"),
    include_str!("../sql/tpch/answers-sf0.01/q11.csv"),
    include_str!("../sql/tpch/answers-sf0.01/q12.csv"),
    include_str!("../sql/tpch/answers-sf0.01/q13.csv"),
    include_str!("../sql/tpch/answers-sf0.01/q14.csv"),
    include_str!("../sql/tpch/answers-sf0.01/q15.csv"),
    include_str!("../sql/tpch/answers-sf0.01/q16.csv"),
    include_str!("../sql/tpch/answers-sf0.01/q17.csv"),
    include_str!("../sql/tpch/answers-sf0.01/q18.csv"),
    include_str!("../sql/tpch/answers-sf0.01/q19.csv"),
    include_str!("../sql/tpch/answers-sf0.01/q20.csv"),
    include_str!("../sql/tpch/answers-sf0.01/q21.csv"),
    include_str!("../sql/tpch/answers-sf0.01/q22.csv"),
];

const TPCH_ANSWERS_SF1: [&str; 22] = [
    include_str!("../sql/tpch/answers-sf1/q01.csv"),
    include_str!("../sql/tpch/answers-sf1/q02.csv"),
    include_str!("../sql/tpch/answers-sf1/q03.csv"),
    include_str!("../sql/tpch/answers-sf1/q04.csv"),
    include_str!("../sql/tpch/answers-sf1/q05.csv"),
    include_str!("../sql/tpch/answers-sf1/q06.csv"),
    include_str!("../sql/tpch/answers-sf1/q07.csv"),
    include_str!("../sql/tpch/answers-sf1/q08.csv"),
    include_str!("../sql/tpch/answers-sf1/q09.csv"),
    include_str!("../sql/tpch/answers-sf1/q10.csv"),
    include_str!("../sql/tpch/answers-sf1/q11.csv"),
    include_str!("../sql/tpch/answers-sf1/q12.csv"),
    include_str!("../sql/tpch/answers-sf1/q13.csv"),
    include_str!("../sql/tpch/answers-sf1/q14.csv"),
    include_str!("../sql/tpch/answers-sf1/q15.csv"),
    include_str!("../sql/tpch/answers-sf1/q16.csv"),
    include_str!("../sql/tpch/answers-sf1/q17.csv"),
    include_str!("../sql/tpch/answers-sf1/q18.csv"),
    include_str!("../sql/tpch/answers-sf1/q19.csv"),
    include_str!("../sql/tpch/answers-sf1/q20.csv"),
    include_str!("../sql/tpch/answers-sf1/q21.csv"),
    include_str!("../sql/tpch/answers-sf1/q22.csv"),
];

/// The eight TPC-H tables, in the order `tabl-generate` writes them.
pub const TPCH_TABLES: [&str; 8] = [
    "region", "nation", "supplier", "customer", "part", "partsupp", "orders", "lineitem",
];

/// One Parquet part and the totals its footer reports.
#[derive(Clone, Debug)]
struct Part {
    path: PathBuf,
    rows: u64,
    bytes: u64,
}

/// A dataset folder: its suite, its tables' parts and its queries.
pub struct Dataset {
    pub suite: Suite,
    pub path: PathBuf,
    tables: Vec<(String, Vec<Part>)>,
    pub queries: Vec<Query>,
}

impl Dataset {
    pub fn open(path: &Path) -> Result<Self> {
        let mut files = Vec::new();
        for entry in std::fs::read_dir(path).map_err(|e| format!("{}: {e}", path.display()))? {
            let path = entry.map_err(|e| e.to_string())?.path();
            if path.extension().is_some_and(|extension| extension == "parquet") {
                files.push(path);
            }
        }
        // Natural order within one table: `part_9` before `part_10`.
        files.sort_by_key(|path| {
            let name = path.file_name().unwrap_or_default().to_string_lossy().into_owned();
            (table_name(&name).to_owned(), name.len(), name)
        });
        let mut tables: Vec<(String, Vec<Part>)> = Vec::new();
        for file in files {
            let name = table_name(&file.file_name().unwrap_or_default().to_string_lossy()).to_owned();
            let part = read_footer(&file)?;
            match tables.last_mut() {
                Some((last, parts)) if *last == name => parts.push(part),
                _ => tables.push((name, vec![part])),
            }
        }
        let has = |name: &str| tables.iter().any(|(table, _)| table == name);
        let suite = if has("trips") {
            Suite::Taxi
        } else if TPCH_TABLES.iter().all(|table| has(table)) {
            Suite::Tpch
        } else if has("hits") {
            Suite::Clickbench
        } else {
            return Err(format!(
                "{} holds neither `trips`, the eight TPC-H tables, nor `hits` as <table>.part_NNN.parquet",
                path.display()
            ));
        };
        let queries = match suite {
            Suite::Taxi => TAXI.iter().map(|(name, sql)| query(name, sql, false)).collect(),
            Suite::Tpch => TPCH
                .iter()
                .enumerate()
                .map(|(index, sql)| query(&format!("q{:02}", index + 1), sql, false))
                .collect(),
            Suite::Clickbench => {
                let file = path.join("queries.sql");
                let text = std::fs::read_to_string(&file)
                    .map_err(|e| format!("{}: {e}; fetch it with tabl-download", file.display()))?;
                text.lines()
                    .filter(|line| !line.trim().is_empty())
                    .enumerate()
                    .map(|(index, sql)| query(&format!("q{index:02}"), sql, sql.contains("LIMIT")))
                    .collect()
            }
            Suite::Documents => return Self::open_documents(path),
        };
        Ok(Self {
            suite,
            path: path.to_path_buf(),
            tables,
            queries,
        })
    }

    /// A folder of key draws, `<count>.arrow` each as `tabl-draw-keys` writes them, as the documents suite: one table
    /// per partition, holding as many rows as the partition holds documents.
    pub fn open_documents(path: &Path) -> Result<Self> {
        let mut draws = Vec::new();
        for entry in std::fs::read_dir(path).map_err(|e| format!("{}: {e}", path.display()))? {
            let file = entry.map_err(|e| e.to_string())?.path();
            let count = file
                .extension()
                .filter(|extension| *extension == "arrow")
                .and(file.file_stem())
                .and_then(|stem| parse_count(&stem.to_string_lossy()));
            if let Some(count) = count {
                draws.push((count, file));
            }
        }
        draws.sort();
        let (_, first) = draws.first().ok_or_else(|| {
            format!(
                "{} holds no key draws such as 1K.arrow; write them with tabl-draw-keys",
                path.display()
            )
        })?;
        let draw = Draw::read(first)?;
        let fields = scalar_fields(&draw.collection).ok_or_else(|| {
            format!(
                "{} draws from `{}`, whose fields are unknown",
                path.display(),
                draw.collection
            )
        })?;
        let tables = draw
            .rows
            .iter()
            .enumerate()
            .map(|(partition, &rows)| {
                let part = Part {
                    path: path.to_path_buf(),
                    rows,
                    bytes: 0,
                };
                (format!("{}_p{partition}", draw.collection), vec![part])
            })
            .collect();
        Ok(Self {
            suite: Suite::Documents,
            path: path.to_path_buf(),
            tables,
            queries: documents_queries(fields, &draws),
        })
    }

    /// Every table, with the fact table cut to the shortest prefix of parts holding at least `rows`.
    pub fn tables(&self, rows: Option<u64>) -> Result<Vec<Table>> {
        let fact = self.suite.fact_table();
        let mut tables = Vec::with_capacity(self.tables.len());
        for (name, parts) in &self.tables {
            let mut count = parts.len();
            if let Some(rows) = rows.filter(|_| name == fact) {
                let available: u64 = parts.iter().map(|part| part.rows).sum();
                if rows > available {
                    return Err(format!("--rows {rows} exceeds the {available} rows of `{name}`"));
                }
                let mut total = 0;
                count = parts
                    .iter()
                    .position(|part| {
                        total += part.rows;
                        total >= rows
                    })
                    .map_or(parts.len(), |index| index + 1);
            }
            let parts = &parts[..count];
            tables.push(Table {
                name: name.clone(),
                parts: parts.iter().map(|part| part.path.clone()).collect(),
                rows: parts.iter().map(|part| part.rows).sum(),
                bytes: parts.iter().map(|part| part.bytes).sum(),
            });
        }
        Ok(tables)
    }

    /// The pinned answer for TPC-H at scale factors 0.01 and 1, judged by the 1.5M orders per unit of scale.
    pub fn expected(&self, query: &Query) -> Option<Answer> {
        if self.suite != Suite::Tpch {
            return None;
        }
        let orders: u64 = self
            .tables
            .iter()
            .find(|(name, _)| name == "orders")?
            .1
            .iter()
            .map(|part| part.rows)
            .sum();
        let index = query.name.strip_prefix('q')?.parse::<usize>().ok()?.checked_sub(1)?;
        let answers = match orders {
            15_000 => &TPCH_ANSWERS_SF0_01,
            1_500_000 => &TPCH_ANSWERS_SF1,
            _ => return None,
        };
        Some(Answer::from_text(answers.get(index)?, '|'))
    }
}

fn query(name: &str, sql: &str, ties: bool) -> Query {
    Query {
        name: name.to_owned(),
        sql: sql.trim().trim_end_matches(';').to_owned(),
        ties,
        documents: None,
    }
}

/// The field prefixes a workload reads, each named `f<count>`, or `all` once it takes every scalar field.
fn field_prefixes(fields: &'static [DocumentField], counts: &[usize]) -> Vec<(String, &'static [DocumentField])> {
    let mut prefixes: Vec<(String, &'static [DocumentField])> = Vec::with_capacity(counts.len());
    for &count in counts {
        let count = count.min(fields.len());
        let name = match count == fields.len() {
            true => "all".to_owned(),
            false => format!("f{count}"),
        };
        if prefixes.iter().all(|(taken, _)| *taken != name) {
            prefixes.push((name, &fields[..count]));
        }
    }
    prefixes
}

/// The queries of the documents suite over the draws `(count, path)`, ascending by count: per draw a keyed gather of
/// one, three and every scalar field and a sample of three, then scans of one and three fields.
fn documents_queries(fields: &'static [DocumentField], draws: &[(u64, PathBuf)]) -> Vec<Query> {
    let mut queries = Vec::new();
    let mut push = |name: String, workload: Workload, keys: &Path, fields: &'static [DocumentField]| {
        queries.push(Query {
            name,
            sql: String::new(),
            // A sample draws other keys on every run, so only its row count can match
            ties: workload == Workload::SampleGather,
            documents: Some(Documents {
                workload,
                keys: keys.to_path_buf(),
                fields,
            }),
        });
    };
    for (count, path) in draws {
        let count = spell_count(*count);
        for (prefix, fields) in field_prefixes(fields, &[1, 3, fields.len()]) {
            push(
                format!("keyed-gather-{count}-{prefix}"),
                Workload::KeyedGather,
                path,
                fields,
            );
        }
        for (prefix, fields) in field_prefixes(fields, &[3]) {
            push(
                format!("sample-gather-{count}-{prefix}"),
                Workload::SampleGather,
                path,
                fields,
            );
        }
    }
    if let Some((_, largest)) = draws.last() {
        for (prefix, fields) in field_prefixes(fields, &[1, 3]) {
            push(format!("scan-{prefix}"), Workload::Scan, largest, fields);
        }
    }
    queries
}

/// `lineitem.part_007.parquet` → `lineitem`, `region.parquet` → `region`.
fn table_name(file: &str) -> &str {
    let stem = file.strip_suffix(".parquet").unwrap_or(file);
    stem.split_once(".part_").map_or(stem, |(name, _)| name)
}

fn read_footer(path: &Path) -> Result<Part> {
    let file = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let bytes = file.metadata().map_err(|e| e.to_string())?.len();
    let reader = SerializedFileReader::new(file).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(Part {
        path: path.to_path_buf(),
        rows: reader.metadata().file_metadata().num_rows() as u64,
        bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_names_drop_part_suffixes() {
        assert_eq!(table_name("lineitem.part_007.parquet"), "lineitem");
        assert_eq!(table_name("region.parquet"), "region");
    }

    /// Writes one Arrow file of `keys` as partition folder `p<partition>` of `alphafold` under `root`.
    fn write_partition(root: &Path, partition: usize, keys: std::ops::Range<u64>) -> PathBuf {
        let folder = root.join("alphafold").join(format!("p{partition}"));
        std::fs::create_dir_all(&folder).unwrap();
        let bytes: Vec<u8> = keys.flat_map(|key| u128::from(key).to_be_bytes()).collect();
        let keys = FixedSizeBinaryArray::try_new(16, bytes.into(), None).unwrap();
        let schema = Arc::new(Schema::new(vec![Field::new(
            "key",
            DataType::FixedSizeBinary(16),
            false,
        )]));
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(keys)]).unwrap();
        let mut writer = IpcFileWriter::try_new(File::create(folder.join("0.arrow")).unwrap(), &schema).unwrap();
        writer.write(&batch).unwrap();
        writer.finish().unwrap();
        folder
    }

    #[test]
    fn draws_repeat_and_keep_each_key_in_its_partition() {
        let root = tempfile::tempdir().unwrap();
        let partitions = [
            write_partition(root.path(), 0, 0..600),
            write_partition(root.path(), 1, 1000..1400),
        ];
        let draws = Draw::sample(&partitions, &[10, 2000]).unwrap();
        assert_eq!(draws, Draw::sample(&partitions, &[10, 2000]).unwrap());
        let [small, whole] = &draws[..] else {
            panic!("two draws")
        };
        assert_eq!(small.rows, [600, 400]);
        assert_eq!(small.keys.iter().map(Vec::len).sum::<usize>(), 10);
        assert_eq!(whole.keys.iter().map(Vec::len).collect::<Vec<_>>(), [600, 400]);
        for (partition, range) in [(0, 0..600u64), (1, 1000..1400)] {
            let mut keys: Vec<u128> = whole.keys[partition]
                .iter()
                .map(|key| u128::from_be_bytes(*key))
                .collect();
            keys.sort_unstable();
            keys.dedup();
            assert_eq!(keys, range.map(u128::from).collect::<Vec<_>>());
        }
        let path = root.path().join("10.arrow");
        small.write(&path).unwrap();
        assert_eq!(&Draw::read(&path).unwrap(), small);
        let dataset = Dataset::open_documents(root.path()).unwrap();
        let names: Vec<&str> = dataset.queries.iter().map(|query| query.name.as_str()).collect();
        assert_eq!(
            names,
            [
                "keyed-gather-10-f1",
                "keyed-gather-10-f3",
                "keyed-gather-10-all",
                "sample-gather-10-f3",
                "scan-f1",
                "scan-f3"
            ]
        );
        assert_eq!(dataset.tables(None).unwrap()[1].name, "alphafold_p1");
    }

    #[test]
    fn documents_digest_alike_across_arrow_types_and_row_order() {
        let batch = |rows: &[usize], integers: ArrayRef, flags: ArrayRef| {
            let keys: Vec<u8> = rows.iter().flat_map(|&row| [row as u8; 16]).collect();
            let names: ArrayRef = Arc::new(arrow::array::StringArray::from_iter(
                rows.iter().map(|&row| (row != 1).then(|| format!("name {row}"))),
            ));
            let schema = Schema::new(vec![
                Field::new("name", DataType::Utf8, true),
                Field::new("count", integers.data_type().clone(), true),
                Field::new("flag", flags.data_type().clone(), true),
            ]);
            keyed_batch(
                keys,
                &RecordBatch::try_new(Arc::new(schema), vec![names, integers, flags]).unwrap(),
            )
            .unwrap()
        };
        let int64: ArrayRef = Arc::new(arrow::array::Int64Array::from(vec![7, 8, 9]));
        let booleans: ArrayRef = Arc::new(arrow::array::BooleanArray::from(vec![true, false, true]));
        let int32: ArrayRef = Arc::new(arrow::array::Int32Array::from(vec![9, 8, 7]));
        let bytes: ArrayRef = Arc::new(UInt8Array::from(vec![1, 0, 1]));
        let mut answers = [Answer::default(), Answer::default()];
        answers[0].push_batches(&[batch(&[0, 1, 2], int64, booleans)]).unwrap();
        answers[1].push_batches(&[batch(&[2, 1, 0], int32, bytes)]).unwrap();
        let [first, second] = &mut answers;
        assert_eq!(first.digest(false), second.digest(false));
    }

    #[test]
    fn tpch_answers_parse() {
        let answer = Answer::from_text(TPCH_ANSWERS_SF1[0], '|');
        assert_eq!(answer.rows(), 4);
    }
}
