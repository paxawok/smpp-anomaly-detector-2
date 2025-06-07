SELECT * FROM smpp_messages 
LIMIT 1000;
DELETE FROM captured_pdus;
DELETE FROM smpp_messages;
DELETE FROM sqlite_sequence WHERE name='captured_pdus';
DELETE FROM sqlite_sequence WHERE name='smpp_messages';