SELECT * FROM smpp_messages 
WHERE recipient_burst = 1
LIMIT 1000;
DELETE FROM captured_pdus;
DELETE FROM smpp_messages;
DELETE FROM sqlite_sequence WHERE name='captured_pdus';
DELETE FROM sqlite_sequence WHERE name='smpp_messages';