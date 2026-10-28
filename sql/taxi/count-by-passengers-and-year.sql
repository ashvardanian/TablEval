SELECT passenger_count, EXTRACT(YEAR FROM pickup_at) AS year, COUNT(*) AS trips
FROM trips
GROUP BY passenger_count, EXTRACT(YEAR FROM pickup_at)
