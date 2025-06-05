"""
SMPP Capture Module - Combined pcap capture and PDU parsing
"""

import struct
import socket
import logging
from datetime import datetime
from typing import Dict, Optional, Tuple, List
from scapy.all import sniff, IP, TCP
import threading
import queue

class SMPPPacketParser:
    """Parser for SMPP protocol packets"""
    
    # SMPP Command IDs
    COMMAND_IDS = {
        0x00000001: 'bind_receiver',
        0x00000002: 'bind_transmitter', 
        0x00000003: 'bind_transceiver',
        0x00000004: 'submit_sm',
        0x00000005: 'deliver_sm',
        0x00000006: 'unbind',
        0x00000007: 'replace_sm',
        0x00000008: 'cancel_sm',
        0x00000009: 'bind_receiver_resp',
        0x80000001: 'bind_receiver_resp',
        0x80000002: 'bind_transmitter_resp',
        0x80000003: 'bind_transceiver_resp',
        0x80000004: 'submit_sm_resp',
        0x80000005: 'deliver_sm_resp',
        0x80000006: 'unbind_resp',
        0x80000007: 'replace_sm_resp',
        0x80000008: 'cancel_sm_resp'
    }
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def parse_header(self, data: bytes) -> Optional[Dict]:
        """Parse SMPP header (16 bytes)"""
        if len(data) < 16:
            return None
            
        try:
            # SMPP header structure
            command_length, command_id, command_status, sequence_number = struct.unpack('!IIII', data[:16])
            
            return {
                'command_length': command_length,
                'command_id': command_id,
                'command_id_name': self.COMMAND_IDS.get(command_id, 'unknown'),
                'command_status': command_status,
                'sequence_number': sequence_number
            }
        except Exception as e:
            self.logger.error(f"Error parsing header: {e}")
            return None
    
    def parse_c_octet_string(self, data: bytes, offset: int) -> Tuple[str, int]:
        """Parse null-terminated string"""
        try:
            null_pos = data.index(b'\x00', offset)
            string_data = data[offset:null_pos]
            
            # Try different encodings
            for encoding in ['utf-8', 'cp1251', 'latin-1']:
                try:
                    text = string_data.decode(encoding)
                    return text, null_pos + 1
                except:
                    continue
                    
            # Fallback to hex representation
            return string_data.hex(), null_pos + 1
            
        except ValueError:
            # No null terminator found
            return "", offset
    
    def parse_submit_sm_body(self, data: bytes) -> Dict:
        """Parse submit_sm PDU body"""
        body = {}
        offset = 16  # Skip header
        
        try:
            # Service type
            body['service_type'], offset = self.parse_c_octet_string(data, offset)
            
            # TON/NPI
            if offset + 2 <= len(data):
                body['source_addr_ton'] = data[offset]
                body['source_addr_npi'] = data[offset + 1]
                offset += 2
            
            # Source address
            body['source_addr'], offset = self.parse_c_octet_string(data, offset)
            
            # Destination TON/NPI
            if offset + 2 <= len(data):
                body['dest_addr_ton'] = data[offset]
                body['dest_addr_npi'] = data[offset + 1]
                offset += 2
            
            # Destination address
            body['dest_addr'], offset = self.parse_c_octet_string(data, offset)
            
            # ESM class, protocol ID, priority flag
            if offset + 3 <= len(data):
                body['esm_class'] = data[offset]
                body['protocol_id'] = data[offset + 1]
                body['priority_flag'] = data[offset + 2]
                offset += 3
            
            # Schedule/validity times (skip)
            _, offset = self.parse_c_octet_string(data, offset)  # schedule_delivery_time
            _, offset = self.parse_c_octet_string(data, offset)  # validity_period
            
            # Message flags
            if offset + 5 <= len(data):
                body['registered_delivery'] = data[offset]
                body['replace_if_present_flag'] = data[offset + 1]
                body['data_coding'] = data[offset + 2]
                body['sm_default_msg_id'] = data[offset + 3]
                body['sm_length'] = data[offset + 4]
                offset += 5
            
            # Message text
            msg_length = body.get('sm_length', 0)
            if msg_length > 0 and offset + msg_length <= len(data):
                msg_data = data[offset:offset + msg_length]
                body['message_text'] = self.decode_message(msg_data, body.get('data_coding', 0))
                body['message_length'] = msg_length
            
            return body
            
        except Exception as e:
            self.logger.error(f"Error parsing submit_sm body: {e}")
            return body
    
    def decode_message(self, data: bytes, data_coding: int) -> str:
        """Decode message based on data_coding"""
        encodings = {
            0: 'latin-1',    # GSM 7-bit
            8: 'utf-16-be',  # UCS2
            4: 'latin-1',    # 8-bit binary
        }
        
        encoding = encodings.get(data_coding, 'utf-8')
        
        try:
            return data.decode(encoding)
        except:
            # Try other encodings
            for enc in ['utf-8', 'cp1251', 'latin-1']:
                try:
                    return data.decode(enc)
                except:
                    continue
            
            # Fallback to hex
            return data.hex()
    
    def parse_pdu(self, data: bytes) -> Optional[Dict]:
        """Parse complete PDU"""
        header = self.parse_header(data)
        if not header:
            return None
        
        pdu = {
            'header': header,
            'raw_data': data.hex(),
            'timestamp': datetime.now()
        }
        
        # Parse body based on command type
        if header['command_id_name'] == 'submit_sm':
            pdu['body'] = self.parse_submit_sm_body(data)
            pdu['category'] = self._detect_category(pdu['body'])
        elif header['command_id_name'] == 'deliver_sm':
            # Similar to submit_sm
            pdu['body'] = self.parse_submit_sm_body(data)
            pdu['category'] = self._detect_category(pdu['body'])
        
        return pdu
    
    def _detect_category(self, body: Dict) -> str:
        """Simple category detection based on content"""
        message = body.get('message_text', '').lower()
        source = body.get('source_addr', '').upper()
        
        # Banking patterns
        if any(word in message for word in ['банк', 'рахунок', 'баланс', 'переказ', 'грн']):
            return 'banking'
        if any(s in source for s in ['PRIVAT', 'MONO', 'OSCHAD', 'PUMB']):
            return 'banking'
            
        # Delivery patterns
        if any(word in message for word in ['доставка', 'посилка', 'замовлення']):
            return 'delivery'
        if any(s in source for s in ['NOVA', 'UKRPOSHTA', 'JUSTIN']):
            return 'delivery'
            
        # OTP patterns
        if any(word in message for word in ['код', 'code', 'пароль', 'password']):
            return 'otp'
            
        return 'other'


class SMPPCapture:
    """Real-time SMPP traffic capture"""
    
    def __init__(self, interface: str = None, port: int = 2775):
        self.interface = interface
        self.port = port
        self.parser = SMPPPacketParser()
        self.logger = logging.getLogger(__name__)
        self.packet_queue = queue.Queue(maxsize=10000)
        self.capture_active = False
        self.stats = {
            'packets_captured': 0,
            'pdus_parsed': 0,
            'parse_errors': 0
        }
        
    def packet_callback(self, packet):
        """Callback for each captured packet"""
        try:
            if IP in packet and TCP in packet:
                tcp_layer = packet[TCP]
                
                # Check if it's SMPP traffic
                if tcp_layer.dport == self.port or tcp_layer.sport == self.port:
                    # Extract packet info
                    packet_info = {
                        'timestamp': datetime.now(),
                        'src_ip': packet[IP].src,
                        'dst_ip': packet[IP].dst,
                        'src_port': tcp_layer.sport,
                        'dst_port': tcp_layer.dport,
                        'payload': bytes(tcp_layer.payload)
                    }
                    
                    # Try to parse as SMPP
                    if len(packet_info['payload']) >= 16:
                        self.packet_queue.put(packet_info)
                        self.stats['packets_captured'] += 1
                        
        except Exception as e:
            self.logger.error(f"Error in packet callback: {e}")
    
    def start_capture(self):
        """Start packet capture"""
        self.capture_active = True
        self.logger.info(f"Starting SMPP capture on interface {self.interface}, port {self.port}")
        
        # Start sniffer in a separate thread
        capture_thread = threading.Thread(
            target=self._capture_thread,
            daemon=True
        )
        capture_thread.start()
        
        # Start processing thread
        process_thread = threading.Thread(
            target=self._process_packets,
            daemon=True
        )
        process_thread.start()
    
    def _capture_thread(self):
        """Capture thread"""
        filter_str = f"tcp port {self.port}"
        
        try:
            sniff(
                iface=self.interface,
                filter=filter_str,
                prn=self.packet_callback,
                store=0,
                stop_filter=lambda x: not self.capture_active
            )
        except Exception as e:
            self.logger.error(f"Capture error: {e}")
    
    def _process_packets(self):
        """Process captured packets"""
        tcp_streams = {}  # Track TCP streams for reassembly
        
        while self.capture_active:
            try:
                # Get packet from queue with timeout
                packet_info = self.packet_queue.get(timeout=1)
                
                # Create stream key
                stream_key = (
                    packet_info['src_ip'], packet_info['src_port'],
                    packet_info['dst_ip'], packet_info['dst_port']
                )
                
                # Simple TCP stream reassembly
                if stream_key not in tcp_streams:
                    tcp_streams[stream_key] = b''
                
                tcp_streams[stream_key] += packet_info['payload']
                
                # Try to parse PDUs from stream
                stream_data = tcp_streams[stream_key]
                
                while len(stream_data) >= 16:
                    # Parse header to get PDU length
                    header = self.parser.parse_header(stream_data)
                    if not header:
                        break
                    
                    pdu_length = header['command_length']
                    
                    if len(stream_data) >= pdu_length:
                        # Extract complete PDU
                        pdu_data = stream_data[:pdu_length]
                        stream_data = stream_data[pdu_length:]
                        
                        # Parse PDU
                        pdu = self.parser.parse_pdu(pdu_data)
                        if pdu:
                            # Add network info
                            pdu['src_ip'] = packet_info['src_ip']
                            pdu['dst_ip'] = packet_info['dst_ip']
                            pdu['src_port'] = packet_info['src_port']
                            pdu['dst_port'] = packet_info['dst_port']
                            
                            self.stats['pdus_parsed'] += 1
                            self.process_pdu(pdu)
                        else:
                            self.stats['parse_errors'] += 1
                    else:
                        # Wait for more data
                        break
                
                # Update stream buffer
                tcp_streams[stream_key] = stream_data
                
                # Clean old streams
                if len(tcp_streams) > 1000:
                    # Remove oldest streams
                    oldest_keys = list(tcp_streams.keys())[:100]
                    for key in oldest_keys:
                        del tcp_streams[key]
                        
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Processing error: {e}")
    
    def process_pdu(self, pdu: Dict):
        """Process parsed PDU - override in subclass"""
        # This is where you would:
        # 1. Save to database
        # 2. Extract features
        # 3. Run anomaly detection
        
        self.logger.debug(f"Processed PDU: {pdu['header']['command_id_name']} "
                         f"from {pdu.get('body', {}).get('source_addr', 'unknown')}")
    
    def stop_capture(self):
        """Stop capture"""
        self.capture_active = False
        self.logger.info("Stopping SMPP capture")
        self.logger.info(f"Capture stats: {self.stats}")
    
    def get_stats(self) -> Dict:
        """Get capture statistics"""
        return self.stats.copy()


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create capture instance
    capture = SMPPCapture(interface="eth0", port=2775)
    
    # Custom PDU processor
    class CustomSMPPCapture(SMPPCapture):
        def process_pdu(self, pdu: Dict):
            # Print PDU info
            if pdu['header']['command_id_name'] in ['submit_sm', 'deliver_sm']:
                body = pdu.get('body', {})
                print(f"SMS from {body.get('source_addr')} to {body.get('dest_addr')}: "
                      f"{body.get('message_text', '')[:50]}...")
    
    # Start capture
    custom_capture = CustomSMPPCapture()
    
    try:
        custom_capture.start_capture()
        # Run for some time
        import time
        time.sleep(60)
    except KeyboardInterrupt:
        pass
    finally:
        custom_capture.stop_capture()
        print(f"Final stats: {custom_capture.get_stats()}")