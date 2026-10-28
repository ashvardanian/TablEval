SELECT passenger_count, EXTRACT(YEAR FROM pickup_at) AS year, FLOOR(trip_distance) AS distance, COUNT(*) AS trips
FROM trips
GROUP BY passenger_count, EXTRACT(YEAR FROM pickup_at), FLOOR(trip_distance)
ORDER BY year, trips DESC, passenger_count, distance
