SELECT vendor_id, COUNT(*) AS trips
FROM trips
GROUP BY vendor_id
