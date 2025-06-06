SELECT id, message_text, timestamp 
FROM smpp_messages 
WHERE message_text IS NOT NULL AND TRIM(message_text) != ''
LIMIT 100;
DELETE FROM captured_pdus;
DELETE FROM smpp_messages;
DELETE FROM sqlite_sequence WHERE name='captured_pdus';
DELETE FROM sqlite_sequence WHERE name='smpp_messages';
SELECT 
    id,
    command_name,
    command_status,
    LENGTH(raw_body) AS body_length,
    HEX(raw_body) AS raw_body_hex,
    raw_body
FROM 
    captured_pdus
WHERE 
    command_name IN ('submit_sm', 'deliver_sm')
    AND LENGTH(raw_body) > 0
ORDER BY 
    timestamp DESC
LIMIT 5;
SELECT 
  id,
  command_name,
  HEX(raw_body) AS raw_body_hex,
  LENGTH(raw_body) AS body_len
FROM captured_pdus
WHERE command_name = 'submit_sm'
  AND INSTR(raw_body, x'08') > 0
LIMIT 20;
