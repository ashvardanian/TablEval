//! UStore's NVMe engine, with one collection of JSON documents per table gathered into Arrow columns per query.
//!
//! The load prints every Parquet row as one document, decimals as their unscaled integers and dates and timestamps as
//! the integers Arrow holds, and lands each table as one ingest. A query lists its table's keys, gathers only the
//! columns it reads, and runs the folds of `folds.rs` over them, so exact runs verify like any other engine's.
//!
//! `--gather cpu` hands the gathered columns to Arrow in place, their releases sharing the arena they lie in.
//! `--gather gpu` gathers on an `Executor::for_vram` executor, whose columns stay in device memory the host folds cannot
//! read, so they cross to the host once through the CUDA driver. It fails when the device has none of the library's
//! kernels, as every gather would otherwise run on the host and report as a GPU run.
//!
//! `--sample-rate` gathers a `docs_sample` draw instead of every key. Its rows are not the table's, so it reports as
//! `ustore-sampled` beside its rate, skips digests, and writes under `sampled/`, away from what `plot.py` compares.
//!
//! Every query reads through one snapshot. A host gather from the NVMe engine splits its key windows across `--threads`
//! lanes, each with an `Executor::for_nvme` on its share of the cores, arenas of its own and a snapshot that must read
//! at the listing's stamp, since a snapshot serves one thread at a time. The engine may pin and cache the whole memory
//! limit, as it reads past the page cache. `--arena ring` hands an arena back once Arrow releases every column lying in
//! it. `--ingest-during-queries` appends `lineitem` once more while `q06` gathers snapshot after snapshot on one lane,
//! failing the query on a snapshot holding no committed prefix. Every query prints the bytes Arrow took over in place
//! and copied for each column.
//!
//! ## Build and run
//!
//! ```sh
//! cargo run --release --no-default-features --features ustore-backend --bin tabl-eval-ustore -- \
//!     --dataset data/tpch-sf1 --gather cpu
//! ```
//!
//! The crate's own CMake build carries no device kernels: `--gather gpu` needs `USTORE_LIBRARY_DIR` to name the `lib/`
//! of a UStore prefix configured with `USTORE_CUDA_ARCH`.

use std::{
    alloc::System,
    collections::{BTreeMap, HashMap},
    ffi::{c_int, c_void, CStr},
    fs::File,
    io::Write,
    mem::{size_of, transmute},
    path::{Path, PathBuf},
    ptr::{null, null_mut},
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc, Arc, Mutex, PoisonError,
    },
    time::Instant,
};

use arrow::{
    array::{
        make_array, Array, ArrayData, ArrayRef, AsArray, FixedSizeBinaryArray, RecordBatch, RecordBatchIterator,
        StringArray,
    },
    buffer::{Buffer, MutableBuffer, OffsetBuffer},
    compute::cast,
    datatypes::{DataType, Date32Type, Decimal128Type, Field, Float64Type, Int64Type, Schema, SchemaRef},
    error::ArrowError,
    ffi::{from_ffi_and_data_type, FFI_ArrowArray},
    ffi_stream::FFI_ArrowArrayStream,
};
use clap::Parser;
use parquet::arrow::arrow_reader::{ParquetRecordBatchReader, ParquetRecordBatchReaderBuilder};
use serde_json::{json, Value};
use ustore::{
    arrow::ArrowDeviceArray, gpu_devices, Arena, ArrowArray, ArrowArrayStream, Binary, CollectionID, Column,
    DataDirectory, Database, DocumentEntry, Durability, Executor, GPUDevice, GPUVendor, Gds, IOOptions, IngestOrder,
    Json, Key, Modality, Modification, NVMEOptions, RocksDBOptions, Snapshot, Text,
};

use tableval::{
    backend::directory_bytes, folds::Folds, output, run, unsupported, Answer, Backend, CommonArgs, Configuration,
    Executed, Query, Result, Source, Table, Verify,
};

/// Rows of one ingested batch and of one gather; each gather holds its columns in an arena of its own.
const BATCH_ROWS: usize = 64 * 1024;

/// Where documents become columns.
#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "kebab-case")]
enum Gather {
    Cpu,
    Gpu,
}

/// The UStore engine the documents live in, both behind the same C ABI.
#[derive(clap::ValueEnum, Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "kebab-case")]
enum Engine {
    #[default]
    Nvme,
    /// Needs the `ustore-rocksdb` feature, and gathers on the host alone.
    Rocksdb,
}

/// In what order a gather reads its keys.
#[derive(clap::ValueEnum, Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "kebab-case")]
enum KeyOrder {
    /// A workaround for the engine reading neighbours one by one, which a caller must choose.
    #[default]
    Spread,
    /// The order a scan lists them, which an application would pass.
    Sorted,
}

/// What becomes of a gather's arena once Arrow releases every column lying in it.
#[derive(clap::ValueEnum, Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "kebab-case")]
enum Arenas {
    /// Freed, so every gather makes a new one.
    PerCall,
    /// Kept for a later gather.
    #[default]
    Ring,
}

#[derive(Parser)]
struct Cli {
    #[command(flatten)]
    common: CommonArgs,
    /// cpu gathers on the host, gpu on the first device.
    #[arg(long, env = "TABLEVAL_GATHER", value_enum, default_value = "cpu")]
    gather: Gather,
    /// spread reads keys READ_STRIDE documents apart, sorted reads them as listed, which the engine serves far slower.
    #[arg(long, env = "TABLEVAL_KEY_ORDER", value_enum, default_value_t)]
    key_order: KeyOrder,
    /// nvme is UStore's own engine; rocksdb serves the same documents and verbs through RocksDB.
    #[arg(long, env = "TABLEVAL_ENGINE", value_enum, default_value_t)]
    engine: Engine,
    /// Gather this share of the keys, a fraction over zero and at most one, as a labelled measurement apart from exact
    /// runs; digests are skipped, and reports land in a `sampled` folder under --output.
    #[arg(long, env = "TABLEVAL_SAMPLE_RATE", value_parser = parse_sample_rate)]
    sample_rate: Option<f64>,
    /// ring keeps a gather's arena for a later one once Arrow released its columns, per-call frees it.
    #[arg(long, env = "TABLEVAL_ARENA", value_enum, default_value_t)]
    arena: Arenas,
    /// Append lineitem once more while q06 gathers snapshot after snapshot, each of which must hold a committed
    /// prefix; digests are skipped, and reports land in an `ingest-during-queries` folder under --output.
    #[arg(long, env = "TABLEVAL_INGEST_DURING_QUERIES", default_value_t = false, value_parser = clap::builder::BoolishValueParser::new())]
    ingest_during_queries: bool,
}

/// How the backend stores and gathers, as the command line sets it and every report records it.
#[derive(Clone, Copy, serde::Serialize)]
struct Settings {
    gather: Gather,
    sample_rate: Option<f64>,
    key_order: KeyOrder,
    engine: Engine,
    arena: Arenas,
    ingest_during_queries: bool,
}

fn parse_sample_rate(text: &str) -> Result<f64> {
    text.parse::<f64>()
        .ok()
        .filter(|rate| *rate > 0.0 && *rate <= 1.0)
        .ok_or_else(|| "expected a fraction over zero and at most one".into())
}

type SetCurrentContext = unsafe extern "C" fn(*mut c_void) -> c_int;
type CopyToHost = unsafe extern "C" fn(*mut c_void, u64, usize) -> c_int;

/// The CUDA driver calls that bring device memory to the host, which no C verb of the library offers.
struct Driver {
    set_current_context: SetCurrentContext,
    copy_to_host: CopyToHost,
    context: *mut c_void,
}

impl Driver {
    /// Resolves the driver the library already loaded, and the primary context of CUDA device `ordinal`.
    fn open(ordinal: u32) -> Result<Self> {
        // SAFETY: the driver's ABI fixes each symbol's signature, and the library initialized the driver.
        unsafe {
            let library = libc::dlopen(c"libcuda.so.1".as_ptr(), libc::RTLD_NOW);
            let symbol = |name: &CStr| match libc::dlsym(library, name.as_ptr()) {
                symbol if symbol.is_null() => Err(format!("the CUDA driver lacks {name:?}")),
                symbol => Ok(symbol),
            };
            let get_device: unsafe extern "C" fn(*mut c_int, c_int) -> c_int = transmute(symbol(c"cuDeviceGet")?);
            let retain_context: unsafe extern "C" fn(*mut *mut c_void, c_int) -> c_int =
                transmute(symbol(c"cuDevicePrimaryCtxRetain")?);
            let (mut device, mut context) = (0, null_mut());
            if get_device(&mut device, ordinal as c_int) != 0 || retain_context(&mut context, device) != 0 {
                return Err(format!("the CUDA driver refused device {ordinal}"));
            }
            Ok(Self {
                set_current_context: transmute::<*mut c_void, SetCurrentContext>(symbol(c"cuCtxSetCurrent")?),
                copy_to_host: transmute::<*mut c_void, CopyToHost>(symbol(c"cuMemcpyDtoH_v2")?),
                context,
            })
        }
    }

    fn download(&self, source: *const u8, bytes: usize) -> Result<Buffer> {
        let mut buffer = MutableBuffer::new(bytes);
        if bytes != 0 {
            // SAFETY: the buffer holds `bytes`, and an exported column names at least as many on the device.
            let status = unsafe { (self.copy_to_host)(buffer.as_mut_ptr().cast(), source as u64, bytes) };
            if status != 0 {
                return Err(format!("the CUDA driver refused a copy with status {status}"));
            }
        }
        // SAFETY: the copy wrote every byte.
        unsafe { buffer.set_len(bytes) };
        Ok(buffer.into())
    }

    /// Copies an exported device column to the host as `data_type`.
    fn column(&self, array: &ArrowArray, data_type: &DataType) -> Result<ArrayRef> {
        // SAFETY: the context outlives the process, and this thread may hold another device's.
        if unsafe { (self.set_current_context)(self.context) } != 0 {
            return Err("the CUDA driver refused the device's context".into());
        }
        let rows = array.length as usize;
        // SAFETY: the host holds the array of buffer addresses, with one entry per buffer of the column's layout.
        let buffer = |index: usize| unsafe { (*array.buffers.add(index)).cast::<u8>() };
        let validity = self.download(buffer(0), rows.div_ceil(8))?;
        let builder = ArrayData::builder(data_type.clone())
            .len(rows)
            .null_bit_buffer(Some(validity));
        let builder = match data_type.primitive_width() {
            Some(width) => builder.add_buffer(self.download(buffer(1), rows * width)?),
            None => {
                let offsets = self.download(buffer(1), (rows + 1) * size_of::<i32>())?;
                let bytes = offsets.typed_data::<i32>()[rows] as usize;
                builder.add_buffer(offsets).add_buffer(self.download(buffer(2), bytes)?)
            }
        };
        // SAFETY: the buffers are the layout of `data_type`, which the gather converted every cell into.
        Ok(make_array(unsafe { builder.build_unchecked() }))
    }
}

/// One loaded table: its collection, its Parquet schema, its rows and the parts they came from.
struct Stored {
    collection: CollectionID,
    schema: SchemaRef,
    rows: u64,
    table: Table,
}

/// Documents apart that one gather reads neighbours, so a ring of reads spans megabytes rather than one flash page.
const READ_STRIDE: usize = 256;

/// `keys` in the order a gather reads them: neighbours share a page, and a full ring of reads of one page runs at a tenth
/// of the drive's rate.
fn spread(keys: &[Key]) -> Vec<Key> {
    (0..READ_STRIDE)
        .flat_map(|first| keys.iter().skip(first).step_by(READ_STRIDE).copied())
        .collect()
}

/// One gather: the fields as the documents convert them, and the Arrow schema those columns land in.
struct Requested {
    columns: Vec<Column<'static>>,
    schema: SchemaRef,
}

impl Requested {
    fn of(names: &[&str], declared: &Schema) -> Result<Self> {
        let mut columns = Vec::with_capacity(names.len());
        let mut fields = Vec::with_capacity(names.len());
        for &name in names {
            let data_type = declared.field_with_name(name).map_err(|e| e.to_string())?.data_type();
            let (column, data_type) = match data_type {
                DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View => {
                    (Column::new::<Binary>(name.to_owned()), DataType::Utf8)
                }
                DataType::Int16 => (Column::new::<i16>(name.to_owned()), DataType::Int16),
                DataType::Int32 | DataType::Date32 => (Column::new::<i32>(name.to_owned()), data_type.clone()),
                DataType::Int64 | DataType::Timestamp(_, _) => (Column::new::<i64>(name.to_owned()), data_type.clone()),
                DataType::Decimal128(_, _) => (Column::new::<i64>(name.to_owned()), DataType::Int64),
                DataType::Float64 => (Column::new::<f64>(name.to_owned()), DataType::Float64),
                other => return Err(unsupported(&format!("no gather into {other}"))),
            };
            columns.push(column);
            fields.push(Field::new(name, data_type, true));
        }
        Ok(Self {
            columns,
            schema: Arc::new(Schema::new(fields)),
        })
    }
}

/// Arenas whose columns were all released, kept for later gathers, or freed at once under `--arena per-call`.
struct ArenaRing {
    kept: Option<Mutex<Vec<Arena>>>,
}

impl ArenaRing {
    fn new(arenas: Arenas) -> Self {
        Self {
            kept: (arenas == Arenas::Ring).then(Mutex::default),
        }
    }

    /// A kept arena, or a new one where `executor` computes, or for the calling thread without one.
    fn take_or_new(&self, executor: Option<&Executor>) -> Result<Arena> {
        let kept = self
            .kept
            .as_ref()
            .and_then(|kept| kept.lock().unwrap_or_else(PoisonError::into_inner).pop());
        match (kept, executor) {
            (Some(arena), _) => Ok(arena),
            (None, Some(executor)) => Arena::new_in(executor).map_err(|e| e.to_string()),
            (None, None) => Arena::new().map_err(|e| e.to_string()),
        }
    }

    fn put(&self, arena: Arena) {
        if let Some(kept) = &self.kept {
            kept.lock().unwrap_or_else(PoisonError::into_inner).push(arena);
        }
    }
}

/// The arena one gather's columns lie in, which the release of the last of them hands back to its ring.
struct HeldArena {
    arena: Option<Arena>,
    ring: Arc<ArenaRing>,
}

impl Drop for HeldArena {
    fn drop(&mut self) {
        if let Some(arena) = self.arena.take() {
            self.ring.put(arena);
        }
    }
}

/// One gather lane: the ring its arenas come back to, and the executor its calls run on, or none for the calling
/// thread.
struct Lane {
    // Declared before the executor, so the kept arenas are freed before it drains.
    ring: Arc<ArenaRing>,
    executor: Option<Executor>,
}

/// Bytes of one column that Arrow took over where the export left them, and bytes it copied.
#[derive(Clone, Copy, Default)]
struct ImportedBytes {
    in_place: u64,
    copied: u64,
}

impl ImportedBytes {
    /// Compares every buffer of `data` with the address the export gave it; a validity bitmap Arrow dropped for
    /// holding no nulls counts as neither.
    fn add(&mut self, data: &ArrayData, exported: &[*const u8]) {
        let validity = data.nulls().map(|nulls| nulls.buffer());
        let buffers = std::iter::once(validity).chain(data.buffers().iter().map(Some));
        for (buffer, &address) in buffers.zip(exported) {
            let Some(buffer) = buffer else { continue };
            match buffer.as_ptr() == address {
                true => self.in_place += buffer.len() as u64,
                false => self.copied += buffer.len() as u64,
            }
        }
    }
}

/// The rows a snapshot of an appended table holds, and the order-free digest of their keys and gathered columns.
#[derive(Clone, Copy, Default, PartialEq, Eq)]
struct Prefix {
    rows: u64,
    digest: u64,
}

/// The finalizer of SplitMix64, spreading every input bit over the output.
fn mix(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

/// The integer a gathered or Parquet cell of a `q06` column holds: a date's days, or a decimal's unscaled value.
fn integer_cell(column: &ArrayRef, row: usize) -> i64 {
    match column.data_type() {
        DataType::Date32 => column.as_primitive::<Date32Type>().value(row).into(),
        DataType::Decimal128(_, _) => column.as_primitive::<Decimal128Type>().value(row) as i64,
        _ => column.as_primitive::<Int64Type>().value(row),
    }
}

/// The sum of a hash of every key with its row of `columns`, so the digest of appended batches adds up to theirs.
fn rows_digest(keys: &[Key], columns: &[ArrayRef]) -> u64 {
    keys.iter().enumerate().fold(0u64, |digest, (row, key)| {
        let key = u128::from_le_bytes(key.0 .0);
        let hash = columns
            .iter()
            .fold(mix(key as u64 ^ (key >> 64) as u64), |hash, column| {
                mix(hash ^ integer_cell(column, row) as u64)
            });
        digest.wrapping_add(hash)
    })
}

/// What every gather of one query shares.
struct Gathering<'requested> {
    collection: CollectionID,
    requested: &'requested Requested,
    /// Whether rows add to the digest of the prefix gathered, which only `--ingest-during-queries` reads.
    digests: bool,
}

/// What one query gathered: its batches tagged with their window, the prefix they hold, and the bytes Arrow took
/// for each column.
struct Gathered {
    batches: Vec<(usize, RecordBatch)>,
    seen: Prefix,
    imported: Vec<ImportedBytes>,
}

impl Gathered {
    fn new(columns: usize) -> Self {
        Self {
            batches: Vec::new(),
            seen: Prefix::default(),
            imported: vec![ImportedBytes::default(); columns],
        }
    }

    fn merge(&mut self, other: Self) {
        self.batches.extend(other.batches);
        self.seen.rows += other.seen.rows;
        self.seen.digest = self.seen.digest.wrapping_add(other.seen.digest);
        for (bytes, theirs) in self.imported.iter_mut().zip(other.imported) {
            bytes.in_place += theirs.in_place;
            bytes.copied += theirs.copied;
        }
    }

    /// The batches in the order their windows were listed.
    fn into_batches(mut self) -> Vec<RecordBatch> {
        self.batches.sort_unstable_by_key(|(window, _)| *window);
        self.batches.into_iter().map(|(_, batch)| batch).collect()
    }
}

/// The bytes Arrow took over in place and copied for each column of one query, as one indented line.
fn spell_imported(requested: &Requested, imported: &[ImportedBytes]) -> String {
    let columns: Vec<String> = requested
        .schema
        .fields()
        .iter()
        .zip(imported)
        .map(|(field, bytes)| format!("{} {}/{}", field.name(), bytes.in_place, bytes.copied))
        .collect();
    format!("  bytes in place/copied into Arrow: {}", columns.join(", "))
}

/// What a host column's release frees after the library's own: its share of the arena its buffers lie in.
struct HeldColumn {
    release: Option<unsafe extern "C" fn(*mut ArrowArray)>,
    private_data: *mut c_void,
    arena: Arc<Mutex<HeldArena>>,
}

unsafe extern "C" fn release_held_column(array: *mut ArrowArray) {
    let held = Box::from_raw((*array).private_data.cast::<HeldColumn>());
    (*array).release = held.release;
    (*array).private_data = held.private_data;
    if let Some(release) = held.release {
        release(array);
    }
    drop(held.arena);
}

/// The table a hand-written query reads and the columns its fold touches.
fn columns_of(query: &str) -> Option<(&'static str, &'static [&'static str])> {
    Some(match query {
        "count-by-vendor" => ("trips", &["vendor_id"]),
        "average-amount-by-passengers" => ("trips", &["passenger_count", "total_amount"]),
        "count-by-passengers-and-year" => ("trips", &["passenger_count", "pickup_at"]),
        "count-by-passengers-year-and-distance" => ("trips", &["passenger_count", "pickup_at", "trip_distance"]),
        "q01" => (
            "lineitem",
            &[
                "l_shipdate",
                "l_returnflag",
                "l_linestatus",
                "l_quantity",
                "l_extendedprice",
                "l_discount",
                "l_tax",
            ],
        ),
        "q06" => (
            "lineitem",
            &["l_shipdate", "l_quantity", "l_extendedprice", "l_discount"],
        ),
        _ => return None,
    })
}

/// The `key` and `value` columns of `batch`: keys counting from `first_key`, and every row printed as one document.
fn documents(batch: &RecordBatch, first_key: u64) -> std::result::Result<RecordBatch, ArrowError> {
    let schema = batch.schema();
    let mut names = Vec::with_capacity(batch.num_columns());
    let mut columns = Vec::with_capacity(batch.num_columns());
    for (field, column) in schema.fields().iter().zip(batch.columns()) {
        names.push(format!("{}:", json!(field.name())));
        columns.push(match column.data_type() {
            DataType::Decimal128(_, _) => column.clone(),
            DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View => cast(column, &DataType::Utf8)?,
            data_type if data_type.is_floating() => cast(column, &DataType::Float64)?,
            _ => cast(column, &DataType::Int64)?,
        });
    }
    let mut text = Vec::new();
    let mut offsets = Vec::with_capacity(batch.num_rows() + 1);
    offsets.push(0);
    for row in 0..batch.num_rows() {
        text.push(b'{');
        for (name, column) in names.iter().zip(&columns) {
            if column.is_null(row) {
                continue;
            }
            text.extend_from_slice(name.as_bytes());
            match column.data_type() {
                DataType::Decimal128(_, _) => write!(text, "{}", column.as_primitive::<Decimal128Type>().value(row))?,
                DataType::Float64 => write!(text, "{}", column.as_primitive::<Float64Type>().value(row))?,
                DataType::Utf8 => serde_json::to_writer(&mut text, column.as_string::<i32>().value(row))
                    .map_err(|e| ArrowError::JsonError(e.to_string()))?,
                _ => write!(text, "{}", column.as_primitive::<Int64Type>().value(row))?,
            }
            text.push(b',');
        }
        if text.last() == Some(&b',') {
            text.pop();
        }
        text.push(b'}');
        offsets.push(text.len() as i32);
    }
    let keys: Vec<u8> = (first_key..first_key + batch.num_rows() as u64)
        .flat_map(|key| Key::from(key).0 .0)
        .collect();
    let keys = FixedSizeBinaryArray::try_new(16, Buffer::from_vec(keys), None)?;
    let values = StringArray::try_new(OffsetBuffer::new(offsets.into()), Buffer::from_vec(text), None)?;
    RecordBatch::try_new(documents_schema(), vec![Arc::new(keys), Arc::new(values)])
}

fn documents_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("key", DataType::FixedSizeBinary(16), false),
        Field::new("value", DataType::Utf8, false),
    ]))
}

/// The rows of `table` as Parquet batches, its parts in order.
fn parquet_batches(table: &Table) -> Result<impl Iterator<Item = std::result::Result<RecordBatch, ArrowError>>> {
    let readers = table
        .parts
        .iter()
        .map(|part| {
            let file = File::open(part).map_err(|e| format!("{}: {e}", part.display()))?;
            ParquetRecordBatchReaderBuilder::try_new(file)
                .and_then(|builder| builder.with_batch_size(BATCH_ROWS).build())
                .map_err(|e| format!("{}: {e}", part.display()))
        })
        .collect::<Result<Vec<ParquetRecordBatchReader>>>()?;
    Ok(readers.into_iter().flatten())
}

/// The parts of `table` as batches of documents, keyed in ascending order from zero.
fn document_batches(table: &Table) -> Result<impl Iterator<Item = std::result::Result<RecordBatch, ArrowError>>> {
    let mut next_key = 0;
    Ok(parquet_batches(table)?.map(move |batch| {
        let batch = batch?;
        let printed = documents(&batch, next_key)?;
        next_key += batch.num_rows() as u64;
        Ok(printed)
    }))
}

/// Batches of documents as one stream, which an ingest takes over.
fn stream_of(
    batches: impl Iterator<Item = std::result::Result<RecordBatch, ArrowError>> + Send + 'static,
) -> ArrowArrayStream {
    let mut stream = ArrowArrayStream::empty();
    let reader = RecordBatchIterator::new(batches, documents_schema());
    // SAFETY: both types are `struct ArrowArrayStream`, and an empty stream holds nothing to overwrite.
    unsafe {
        (&raw mut stream)
            .cast::<FFI_ArrowArrayStream>()
            .write(FFI_ArrowArrayStream::new(Box::new(reader)))
    };
    stream
}

fn table_schema(table: &Table) -> Result<SchemaRef> {
    let part = table
        .parts
        .first()
        .ok_or_else(|| format!("{} has no parts", table.name))?;
    let file = File::open(part).map_err(|e| format!("{}: {e}", part.display()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).map_err(|e| e.to_string())?;
    Ok(builder.schema().clone())
}

struct UstoreBackend {
    // Declared first, so the lanes and their arenas go before the database, which closes before its opening executor.
    lanes: Vec<Lane>,
    listings: ArenaRing,
    database: Option<Database>,
    /// The executor the database opens on when lanes share its I/O channels.
    opening: Option<Executor>,
    /// Executor threads of each lane, zero when the one lane gathers on the calling thread or a device.
    lane_threads: usize,
    device: Option<(Driver, GPUDevice)>,
    folds: Folds,
    tables: HashMap<String, Stored>,
    settings: Settings,
    path: PathBuf,
    /// Data directories after the manifest's own, each on a drive of its own.
    further: Vec<PathBuf>,
}

/// The database whose manifest lies in `path`, spreading its files over `further`, opened on `opening` or the calling
/// thread.
fn open_database(engine: Engine, path: &Path, further: &[PathBuf], opening: Option<&Executor>) -> Result<Database> {
    let manifest = path.join("tableval.ustore.json");
    let manifest = manifest.to_str().ok_or("the data directory is not UTF-8")?;
    let directory = path.to_str().ok_or("the data directory is not UTF-8")?;
    let data_directories = further
        .iter()
        .map(|folder| folder.to_str().map(|path| DataDirectory { path, max_bytes: 0 }))
        .collect::<Option<Vec<_>>>()
        .ok_or("a data directory is not UTF-8")?;
    match engine {
        Engine::Nvme => Database::open_nvme(&NVMEOptions {
            data_directories: &data_directories,
            executor: opening,
            memory_limit: output::budget().memory_limit,
            ..NVMEOptions::at(manifest)
        }),
        Engine::Rocksdb => Database::open_rocksdb(&RocksDBOptions {
            data_directories: &data_directories,
            ..RocksDBOptions::at(directory)
        }),
    }
    .map_err(|e| e.to_string())
}

impl UstoreBackend {
    fn open(configuration: &Configuration, path: &Path, further: Vec<PathBuf>, settings: Settings) -> Result<Self> {
        if settings.engine == Engine::Rocksdb && settings.gather == Gather::Gpu {
            return Err("--gather gpu needs --engine nvme, as RocksDB gathers on the host alone".into());
        }
        if settings.ingest_during_queries && (settings.engine != Engine::Nvme || settings.sample_rate.is_some()) {
            return Err("--ingest-during-queries needs --engine nvme and no --sample-rate".into());
        }
        // Each lane pins a snapshot of its own, so an ingest racing the gathers keeps to one.
        let lanes_count = match settings.engine == Engine::Nvme && !settings.ingest_during_queries {
            true => configuration.threads.get(),
            false => 1,
        };
        let ring = || Arc::new(ArenaRing::new(settings.arena));
        let (lanes, lane_threads, device, opening) = match settings.gather {
            Gather::Gpu => {
                let device = first_device()?.ok_or("--gather gpu found no device")?;
                if device.vendor != GPUVendor::Nvidia {
                    return Err("--gather gpu reads columns back through the CUDA driver alone".into());
                }
                // SAFETY: a null stream lends nothing, so the executor creates its own.
                let executor = unsafe { Executor::for_vram(device.ordinal as i32, null_mut(), 0, Gds::Automatic) }
                    .map_err(|e| e.to_string())?;
                let driver = Driver::open(device.vendor_ordinal)?;
                let lane = Lane {
                    ring: ring(),
                    executor: Some(executor),
                };
                (vec![lane], 0, Some((driver, device)), None)
            }
            Gather::Cpu if lanes_count == 1 => {
                let lane = Lane {
                    ring: ring(),
                    executor: None,
                };
                (vec![lane], 0, None, None)
            }
            Gather::Cpu => {
                let threads = (output::budget().cores / lanes_count).max(1);
                // Threads outside the opening executor share these channels, so every lane's thread gets one.
                let io = IOOptions {
                    shared_io_channels: lanes_count * threads,
                    ..IOOptions::default()
                };
                let opening = Executor::for_nvme(1, &io).map_err(|e| e.to_string())?;
                let lanes = (0..lanes_count)
                    .map(|_| -> Result<Lane> {
                        let executor = Executor::for_nvme(threads, &IOOptions::default()).map_err(|e| e.to_string())?;
                        Ok(Lane {
                            ring: ring(),
                            executor: Some(executor),
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                (lanes, threads, None, Some(opening))
            }
        };
        let database = open_database(settings.engine, path, &further, opening.as_ref())?;
        Ok(Self {
            lanes,
            listings: ArenaRing::new(settings.arena),
            database: Some(database),
            opening,
            lane_threads,
            device,
            folds: Folds::open(configuration.threads.get())?,
            tables: HashMap::new(),
            settings,
            path: path.to_path_buf(),
            further,
        })
    }

    fn database(&self) -> Result<&Database> {
        self.database
            .as_ref()
            .ok_or_else(|| "the database failed to reopen".into())
    }

    /// Writes the documents of `table` into `collection` a batch at a time, under the keys an ingest gives them.
    fn write_documents(&self, collection: CollectionID, table: &Table) -> Result<usize> {
        let head = self.database()?.head();
        let mut rows = 0;
        for batch in document_batches(table)? {
            let batch = batch.map_err(|e| e.to_string())?;
            let texts = batch
                .column(1)
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or("documents print as UTF-8 text")?;
            let entries: Vec<DocumentEntry<'_, Json>> = (0..texts.len())
                .map(|row| DocumentEntry::new(Key::from((rows + row) as u64), texts.value(row)))
                .collect();
            head.docs_write(
                collection,
                entries.as_slice(),
                Modification::Upsert,
                Durability::Buffered,
            )
            .map_err(|e| format!("{}: {e}", table.name))?;
            rows += texts.len();
        }
        Ok(rows)
    }

    /// Hands `each` the keys of every gather of `stored` in the order it reads them, as `snapshot` lists or samples
    /// them.
    fn for_each_window(
        &self,
        snapshot: &Snapshot<'_>,
        stored: &Stored,
        mut each: impl FnMut(&[Key]) -> Result<()>,
    ) -> Result<()> {
        let mut listing = self.listings.take_or_new(None)?;
        let view = snapshot.view();
        match self.settings.sample_rate {
            Some(rate) => {
                let wanted = (stored.rows as f64 * rate).ceil() as usize;
                let keys = view
                    .docs_sample(stored.collection, wanted, &mut listing)
                    .map_err(|e| e.to_string())?;
                for keys in keys.chunks(BATCH_ROWS) {
                    each(keys)?;
                }
            }
            None => {
                let mut start = Key::from(0u64);
                loop {
                    let keys = view
                        .docs_scan(stored.collection, start, BATCH_ROWS, &mut listing)
                        .map_err(|e| e.to_string())?;
                    let Some(last) = keys.last() else { break };
                    start = last.next();
                    match self.settings.key_order {
                        KeyOrder::Spread => each(&spread(keys))?,
                        KeyOrder::Sorted => each(keys)?,
                    }
                }
            }
        }
        self.listings.put(listing);
        Ok(())
    }

    /// The columns of every key of `stored` as `snapshot` holds them, or of a `docs_sample` draw when the run samples,
    /// with the windows split across the lanes.
    fn gather(&self, snapshot: &Snapshot<'_>, stored: &Stored, requested: &Requested) -> Result<Gathered> {
        let gathering = Gathering {
            collection: stored.collection,
            requested,
            digests: self.settings.ingest_during_queries,
        };
        let columns = requested.columns.len();
        if let [lane] = self.lanes.as_slice() {
            let driver = self.device.as_ref().map(|(driver, _)| driver);
            let mut gathered = Gathered::new(columns);
            let mut window = 0;
            self.for_each_window(snapshot, stored, |keys| {
                gather_window(lane, snapshot, driver, &gathering, window, keys, &mut gathered)?;
                window += 1;
                Ok(())
            })?;
            return Ok(gathered);
        }
        let database = self.database()?;
        let read_stamp = snapshot.read_stamp();
        let (sender, receiver) = mpsc::sync_channel::<(usize, Vec<Key>)>(self.lanes.len());
        // The lanes share the receiver, which goes with the last of them, so the listing stops once every lane has.
        let receiver = Arc::new(Mutex::new(receiver));
        std::thread::scope(|scope| {
            let gathering = &gathering;
            let lanes: Vec<_> = self
                .lanes
                .iter()
                .map(|lane| {
                    let receiver = receiver.clone();
                    scope.spawn(move || -> Result<Gathered> {
                        let snapshot = database.snapshot().map_err(|e| e.to_string())?;
                        if snapshot.read_stamp() != read_stamp {
                            return Err("a lane's snapshot reads at another stamp than the listing's".to_owned());
                        }
                        let mut gathered = Gathered::new(columns);
                        loop {
                            let next = receiver.lock().unwrap_or_else(PoisonError::into_inner).recv();
                            let Ok((window, keys)) = next else { break };
                            gather_window(lane, &snapshot, None, gathering, window, &keys, &mut gathered)?;
                        }
                        Ok(gathered)
                    })
                })
                .collect();
            drop(receiver);
            let mut window = 0;
            let listed = self.for_each_window(snapshot, stored, |keys| {
                sender
                    .send((window, keys.to_vec()))
                    .map_err(|_| "every gather lane stopped".to_owned())?;
                window += 1;
                Ok(())
            });
            drop(sender);
            let mut gathered = Gathered::new(columns);
            for lane in lanes {
                gathered.merge(lane.join().map_err(|_| "a gather lane panicked".to_owned())??);
            }
            listed.map(|()| gathered)
        })
    }

    /// Gathers snapshot after snapshot while one writer appends the rows of `stored` once more, each snapshot holding
    /// a prefix the writer committed, and answers the last, taken once the writer finished.
    fn gather_during_ingest(&self, stored: &Stored, names: &[&str], requested: &Requested) -> Result<Gathered> {
        let database = self.database()?;
        let before = {
            let snapshot = database.snapshot().map_err(|e| e.to_string())?;
            self.gather(&snapshot, stored, requested)?.seen
        };
        if before.rows != stored.rows {
            return Err(format!(
                "gathered {} of {} keys before the writer started",
                before.rows, stored.rows
            ));
        }
        let prefixes = Mutex::new(vec![before]);
        let stop = AtomicBool::new(false);
        std::thread::scope(|scope| {
            let writer = scope.spawn(|| append(database, &stored.table, names, before, &prefixes, &stop));
            let mut snapshots = 0;
            let checked = loop {
                let finished = writer.is_finished();
                let gathered = database
                    .snapshot()
                    .map_err(|e| e.to_string())
                    .and_then(|snapshot| self.gather(&snapshot, stored, requested));
                let gathered = match gathered {
                    Ok(gathered) => gathered,
                    Err(failure) => break Err(failure),
                };
                snapshots += 1;
                let committed = prefixes.lock().unwrap_or_else(PoisonError::into_inner);
                let held = match finished {
                    true => committed.last() == Some(&gathered.seen),
                    false => committed.contains(&gathered.seen),
                };
                if !held {
                    break Err(format!(
                        "snapshot {snapshots} holds {} rows with digest {:016x}, no prefix of the {} the writer listed",
                        gathered.seen.rows,
                        gathered.seen.digest,
                        committed.len()
                    ));
                }
                drop(committed);
                if finished {
                    break Ok(gathered);
                }
            };
            stop.store(true, Ordering::Relaxed);
            writer.join().map_err(|_| "the writer panicked".to_owned())??;
            let gathered = checked?;
            eprintln!(
                "  {snapshots} snapshots during the ingest, each holding a committed prefix, the last all {} rows",
                gathered.seen.rows
            );
            Ok(gathered)
        })
    }

    /// Refuses a device without the library's kernels, where gathers fall to the host silently and only a text export fails.
    fn check_kernels(&self, collection: CollectionID) -> Result<()> {
        let Some((driver, _)) = &self.device else {
            return Ok(());
        };
        let text = Requested {
            columns: vec![Column::new::<Text>("text")],
            schema: Arc::new(Schema::new(vec![Field::new("text", DataType::Utf8, true)])),
        };
        let snapshot = self.database()?.snapshot().map_err(|e| e.to_string())?;
        let mut imported = [ImportedBytes::default()];
        let gathered = gather_keys(
            &self.lanes[0],
            &snapshot,
            Some(driver),
            collection,
            &[Key::from(0u64)],
            &text,
            &mut imported,
        );
        match gathered {
            Ok(_) => Ok(()),
            Err(failure) => Err(format!(
                "--gather gpu needs a library with device kernels, named by USTORE_LIBRARY_DIR: {failure}"
            )),
        }
    }
}

/// Gathers `keys`, the window listed `window`-th, on `lane` into `gathered`.
fn gather_window(
    lane: &Lane,
    snapshot: &Snapshot<'_>,
    driver: Option<&Driver>,
    gathering: &Gathering<'_>,
    window: usize,
    keys: &[Key],
    gathered: &mut Gathered,
) -> Result<()> {
    let batch = gather_keys(
        lane,
        snapshot,
        driver,
        gathering.collection,
        keys,
        gathering.requested,
        &mut gathered.imported,
    )?;
    if gathering.digests {
        gathered.seen.digest = gathered.seen.digest.wrapping_add(rows_digest(keys, batch.columns()));
    }
    gathered.seen.rows += keys.len() as u64;
    gathered.batches.push((window, batch));
    Ok(())
}

/// The columns of `keys` as one batch: the arena on host pages reaches Arrow in place, the device's by copy through
/// `driver`, and `imported` adds up the bytes of each column either way.
fn gather_keys(
    lane: &Lane,
    snapshot: &Snapshot<'_>,
    driver: Option<&Driver>,
    collection: CollectionID,
    keys: &[Key],
    requested: &Requested,
    imported: &mut [ImportedBytes],
) -> Result<RecordBatch> {
    let mut arena = lane.ring.take_or_new(lane.executor.as_ref())?;
    let view = match &lane.executor {
        Some(executor) => snapshot.on(executor),
        None => snapshot.view(),
    };
    let gathered = view
        .docs_gather(collection, keys, &requested.columns, &mut arena)
        .map_err(|e| e.to_string())?;
    let exported = gathered.to_arrow().map_err(|e| e.to_string())?;
    // SAFETY: the arena moves into every host column's release below, and outlives every device column's copy.
    let columns: Vec<ArrowDeviceArray> = exported
        .into_iter()
        .map(|export| unsafe { export.into_parts() }.1)
        .collect();
    // The mutex only lets the columns, released on any thread, share the arena.
    let arena = Arc::new(Mutex::new(HeldArena {
        arena: Some(arena),
        ring: lane.ring.clone(),
    }));
    let mut arrays = Vec::with_capacity(columns.len());
    for ((mut array, field), bytes) in columns.into_iter().zip(requested.schema.fields()).zip(imported) {
        arrays.push(import(&mut array, field.data_type(), &arena, driver, bytes)?);
    }
    RecordBatch::try_new(requested.schema.clone(), arrays).map_err(|e| e.to_string())
}

/// One exported column as Arrow: host memory taken over in place with a share of `arena`, and device memory copied
/// down through `driver`, its bytes added to `bytes` as taken in place or copied.
fn import(
    exported: &mut ArrowDeviceArray,
    data_type: &DataType,
    arena: &Arc<Mutex<HeldArena>>,
    driver: Option<&Driver>,
    bytes: &mut ImportedBytes,
) -> Result<ArrayRef> {
    if exported.device_type == ArrowDeviceArray::CPU {
        let held = Box::new(HeldColumn {
            release: exported.array.release,
            private_data: exported.array.private_data,
            arena: arena.clone(),
        });
        exported.array.private_data = Box::into_raw(held).cast();
        exported.array.release = Some(release_held_column);
        let mut addresses = [null::<u8>(); 3];
        // SAFETY: both types are `struct ArrowArray`, the export laid the column out as `data_type`, and its array of
        // buffer addresses holds `n_buffers` entries, at most three for the types a gather converts into.
        let data = unsafe {
            let count = exported.array.n_buffers as usize;
            for (index, address) in addresses.iter_mut().enumerate().take(count) {
                *address = (*exported.array.buffers.add(index)).cast::<u8>();
            }
            let array = FFI_ArrowArray::from_raw((&raw mut exported.array).cast());
            from_ffi_and_data_type(array, data_type.clone())
        }
        .map_err(|e| e.to_string())?;
        bytes.add(&data, &addresses);
        return Ok(make_array(data));
    }
    let driver = driver.ok_or("a host gather exported a device column")?;
    let array = driver.column(&exported.array, data_type)?;
    let data = array.to_data();
    let validity = data.nulls().map_or(0, |nulls| nulls.buffer().len());
    bytes.copied += (validity + data.buffers().iter().map(Buffer::len).sum::<usize>()) as u64;
    Ok(array)
}

/// Appends the rows of `table` once more past `before`, one ingest per batch, listing each prefix before its ingest
/// lands, until the pass ends or `stop` is raised.
fn append(
    database: &Database,
    table: &Table,
    names: &[&str],
    before: Prefix,
    prefixes: &Mutex<Vec<Prefix>>,
    stop: &AtomicBool,
) -> Result<()> {
    let mut collection = database.collection(&table.name).map_err(|e| e.to_string())?;
    let mut prefix = before;
    for batch in parquet_batches(table)? {
        if stop.load(Ordering::Relaxed) {
            break;
        }
        let batch = batch.map_err(|e| e.to_string())?;
        let keys: Vec<Key> = (prefix.rows..prefix.rows + batch.num_rows() as u64)
            .map(Key::from)
            .collect();
        let columns = names
            .iter()
            .map(|name| {
                let column = batch.column_by_name(name).cloned();
                column.ok_or_else(|| format!("{} lacks {name}", table.name))
            })
            .collect::<Result<Vec<_>>>()?;
        let printed = documents(&batch, prefix.rows).map_err(|e| e.to_string())?;
        prefix = Prefix {
            rows: prefix.rows + keys.len() as u64,
            digest: prefix.digest.wrapping_add(rows_digest(&keys, &columns)),
        };
        prefixes.lock().unwrap_or_else(PoisonError::into_inner).push(prefix);
        collection
            .ingest_arrow(
                stream_of(std::iter::once(Ok(printed))),
                IngestOrder::Sorted,
                Durability::Buffered,
            )
            .map_err(|e| format!("{}: {e}", table.name))?;
    }
    Ok(())
}

impl Backend for UstoreBackend {
    fn metadata(&self) -> BTreeMap<String, Value> {
        let backend = match self.settings.sample_rate {
            Some(_) => "ustore-sampled",
            None => "ustore",
        };
        let exact = self.settings.sample_rate.is_none() && !self.settings.ingest_during_queries;
        BTreeMap::from([
            ("backend".into(), json!(backend)),
            ("library_version".into(), json!(ustore::version())),
            ("engine".into(), json!(self.settings.engine)),
            ("threads".into(), json!(self.folds.threads_count())),
            (
                "gather".into(),
                json!(match self.device {
                    Some(_) => "gpu",
                    None => "cpu",
                }),
            ),
            (
                "gather_device".into(),
                json!(self.device.as_ref().map(|(_, device)| &device.name)),
            ),
            ("sampling_rate".into(), json!(self.settings.sample_rate)),
            ("key_order".into(), json!(self.settings.key_order)),
            ("exact".into(), json!(exact)),
            ("arena".into(), json!(self.settings.arena)),
            ("gather_lanes".into(), json!(self.lanes.len())),
            ("lane_executor_threads".into(), json!(self.lane_threads)),
            ("data_directories".into(), json!(1 + self.further.len())),
            (
                "ingest_during_queries".into(),
                json!(self.settings.ingest_during_queries),
            ),
        ])
    }

    fn load(&mut self, tables: &[Table], source: Source) -> Result<()> {
        if source == Source::Parquet {
            return Err(unsupported("UStore gathers from the documents it ingested"));
        }
        for table in tables {
            let schema = table_schema(table)?;
            let database = self.database()?;
            let collection = database
                .collection_create(&table.name, Modality::Docs)
                .map_err(|e| e.to_string())?;
            self.check_kernels(collection)?;
            let rows = match self.settings.engine {
                Engine::Nvme => {
                    let stream = stream_of(document_batches(table)?);
                    database
                        .collection(&table.name)
                        .and_then(|mut documents| {
                            documents.ingest_arrow(stream, IngestOrder::Sorted, Durability::Buffered)
                        })
                        .map_err(|e| format!("{}: {e}", table.name))?
                }
                // RocksDB serves no ingest, so the same documents land through batched writes.
                Engine::Rocksdb => self.write_documents(collection, table)?,
            };
            self.tables.insert(
                table.name.clone(),
                Stored {
                    collection,
                    schema,
                    rows: rows as u64,
                    table: table.clone(),
                },
            );
        }
        self.database()?.flush().map_err(|e| e.to_string())?;
        for folder in std::iter::once(&self.path).chain(&self.further) {
            eprintln!("  {}: {} bytes", folder.display(), directory_bytes(folder)?);
        }
        Ok(())
    }

    fn run(&mut self, query: &Query, answer: &mut Answer) -> Result<Executed> {
        let Some((table, names)) = columns_of(&query.name).filter(|(table, _)| self.tables.contains_key(*table)) else {
            return Err(unsupported("not hand-written for UStore"));
        };
        if self.settings.ingest_during_queries && query.name != "q06" {
            return Err(unsupported("--ingest-during-queries runs q06 alone"));
        }
        let stored = &self.tables[table];
        let requested = Requested::of(names, &stored.schema)?;
        let start = Instant::now();
        let gathered = match self.settings.ingest_during_queries {
            true => self.gather_during_ingest(stored, names, &requested)?,
            false => {
                let snapshot = self.database()?.snapshot().map_err(|e| e.to_string())?;
                let gathered = self.gather(&snapshot, stored, &requested)?;
                if self.settings.sample_rate.is_none() && gathered.seen.rows != stored.rows {
                    return Err(format!("gathered {} of {} keys", gathered.seen.rows, stored.rows));
                }
                gathered
            }
        };
        let gathered_seconds = start.elapsed().as_secs_f64();
        eprintln!("{}", spell_imported(&requested, &gathered.imported));
        let rows = gathered.seen.rows;
        self.folds.tables.insert(table.to_owned(), gathered.into_batches());
        let folded = self.folds.run::<Int64Type>(query, answer);
        self.folds.tables.clear();
        if let (true, Some(stored)) = (self.settings.ingest_during_queries, self.tables.get_mut(table)) {
            stored.rows = rows;
        }
        Ok(Executed {
            seconds: gathered_seconds + folded?.seconds,
        })
    }

    fn disk_bytes(&self) -> Result<u64> {
        std::iter::once(&self.path)
            .chain(&self.further)
            .map(|folder| directory_bytes(folder))
            .sum()
    }

    fn multiple_data_dirs(&self) -> bool {
        true
    }

    fn reopen(&mut self) -> Result<()> {
        self.database()?.flush().map_err(|e| e.to_string())?;
        self.database = None;
        let database = open_database(self.settings.engine, &self.path, &self.further, self.opening.as_ref())?;
        self.database = Some(database);
        Ok(())
    }
}

fn first_device() -> Result<Option<GPUDevice>> {
    let devices = gpu_devices().map_err(|e| e.to_string())?;
    Ok(devices.into_iter().next())
}

fn main() -> Result<()> {
    let mut cli: Cli = tableval::parse_cli();
    let gather = cli.gather;
    if cli.sample_rate.is_some() {
        cli.common.verify = Verify::None;
        cli.common.output = cli.common.output.join("sampled");
    }
    if cli.ingest_during_queries {
        cli.common.verify = Verify::None;
        cli.common.output = cli.common.output.join("ingest-during-queries");
    }
    let settings = Settings {
        gather,
        sample_rate: cli.sample_rate,
        key_order: cli.key_order,
        engine: cli.engine,
        arena: cli.arena,
        ingest_during_queries: cli.ingest_during_queries,
    };
    let spelled_rate = settings
        .sample_rate
        .map_or_else(|| "none, every key".to_owned(), |rate| rate.to_string());
    let spelled = [
        ("Gather", tableval::spell_value(&gather)),
        ("Sample rate", spelled_rate),
        ("Key order", tableval::spell_value(&settings.key_order)),
        ("Engine", tableval::spell_value(&settings.engine)),
        ("Arena", tableval::spell_value(&settings.arena)),
        ("Ingest during queries", settings.ingest_during_queries.to_string()),
    ];
    let common = cli.common.clone();
    run(cli.common, settings, &spelled, |configuration, path| {
        Ok(Box::new_in(
            UstoreBackend::open(configuration, path, common.further_data_dirs(path), settings)?,
            System,
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn settings(gather: Gather, sample_rate: Option<f64>, key_order: KeyOrder) -> Settings {
        Settings {
            gather,
            sample_rate,
            key_order,
            engine: Engine::Nvme,
            arena: Arenas::Ring,
            ingest_during_queries: false,
        }
    }

    fn open(
        configuration: &tableval::Configuration,
        path: &std::path::Path,
    ) -> tableval::Result<Box<dyn tableval::Backend, System>> {
        Ok(Box::new_in(
            UstoreBackend::open(
                configuration,
                path,
                Vec::new(),
                settings(Gather::Cpu, None, KeyOrder::Sorted),
            )?,
            System,
        ))
    }

    #[test]
    fn taxi_fixture() {
        let root = tempfile::tempdir().unwrap();
        tableval::fixture::assert_taxi_answers(root.path(), open);
    }

    #[test]
    fn reopened_lanes_gather_the_same_rows() {
        let root = tempfile::tempdir().unwrap();
        tableval::fixture::write_taxi_fixture(&root.path().join("taxi")).unwrap();
        let dataset = tableval::Dataset::open(&root.path().join("taxi")).unwrap();
        let storage = root.path().join("storage");
        std::fs::create_dir_all(&storage).unwrap();
        let configuration = Configuration {
            rows: None,
            threads: tableval::Threads(std::num::NonZeroUsize::new(3).unwrap()),
            source: Source::Loaded,
        };
        let settings = settings(Gather::Cpu, None, KeyOrder::Spread);
        let mut backend = UstoreBackend::open(&configuration, &storage, Vec::new(), settings).unwrap();
        backend.load(&dataset.tables(None).unwrap(), Source::Loaded).unwrap();
        let query = &dataset.queries[0];
        let mut answers = [Answer::default(), Answer::default()];
        backend.run(query, &mut answers[0]).unwrap();
        backend.reopen().unwrap();
        backend.run(query, &mut answers[1]).unwrap();
        assert_eq!(backend.lanes.len(), 3);
        assert!(answers[0].rows() > 0);
        assert_eq!(answers[0].digest(query.ties), answers[1].digest(query.ties));
    }

    #[test]
    #[ignore = "needs a GPU and a library with device kernels"]
    fn taxi_fixture_on_device() {
        let root = tempfile::tempdir().unwrap();
        tableval::fixture::assert_taxi_answers(root.path(), |configuration, path| {
            Ok(Box::new_in(
                UstoreBackend::open(
                    configuration,
                    path,
                    Vec::new(),
                    settings(Gather::Gpu, None, KeyOrder::Sorted),
                )?,
                System,
            ))
        });
    }

    #[cfg(feature = "generate")]
    #[test]
    fn tpch_answers() {
        let root = tempfile::tempdir().unwrap();
        tableval::fixture::assert_tpch_answers(root.path(), open);
    }

    #[test]
    fn sampled_runs_never_report_as_exact() {
        let root = tempfile::tempdir().unwrap();
        tableval::fixture::write_taxi_fixture(&root.path().join("taxi")).unwrap();
        let dataset = tableval::Dataset::open(&root.path().join("taxi")).unwrap();
        let storage = root.path().join("storage");
        std::fs::create_dir_all(&storage).unwrap();
        let configuration = Configuration {
            rows: None,
            threads: tableval::Threads(std::num::NonZeroUsize::new(2).unwrap()),
            source: Source::Loaded,
        };
        let settings = settings(Gather::Cpu, Some(0.5), KeyOrder::Spread);
        let mut backend = UstoreBackend::open(&configuration, &storage, Vec::new(), settings).unwrap();
        backend.load(&dataset.tables(None).unwrap(), Source::Loaded).unwrap();
        let mut answer = Answer::default();
        backend.run(&dataset.queries[0], &mut answer).unwrap();
        let metadata = backend.metadata();
        assert_eq!(metadata["backend"], json!("ustore-sampled"));
        assert_eq!(metadata["sampling_rate"], json!(0.5));
        assert_eq!(metadata["exact"], json!(false));
    }
}
