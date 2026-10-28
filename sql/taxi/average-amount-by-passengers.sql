SELECT passenger_count, AVG(total_amount) AS average_amount
FROM trips
GROUP BY passenger_count
