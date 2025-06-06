from smpp_pdu_parser import process_and_store_pdus

count = process_and_store_pdus("data/db/smpp.sqlite", limit=1000)
print(f"[✓] Оброблено та збережено {count} повідомлень")
