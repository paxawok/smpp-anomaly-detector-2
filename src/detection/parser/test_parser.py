from smpp_pdu_parser_2 import process_and_store_pdus, parse_submit_sm

count = process_and_store_pdus("data/db/smpp.sqlite", limit=100000)
print(f"[✓] Оброблено та збережено {count} повідомлень")
