"""
Network Monitor Service
Collects real-time network data from various sources
"""

import asyncio
import subprocess
import psutil
import socket
import struct
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging
import json
import re
from dataclasses import dataclass, asdict
import pandas as pd

from app.core.config import settings
from app.data.collectors.netstat_collector import NetstatCollector
from app.data.collectors.pcap_collector import PcapCollector
from app.data.collectors.log_collector import LogCollector

logger = logging.getLogger(__name__)

@dataclass
class NetworkConnection:
    """Network connection data structure"""
    timestamp: str
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str
    state: str
    process_name: Optional[str] = None
    bytes_sent: int = 0
    bytes_recv: int = 0
    duration: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class NetworkStats:
    """Network statistics data structure"""
    timestamp: str
    total_connections: int
    active_connections: int
    bytes_sent_total: int
    bytes_recv_total: int
    packets_sent: int
    packets_recv: int
    errors_in: int
    errors_out: int
    drops_in: int
    drops_out: int

class NetworkMonitor:
    """Main network monitoring service"""
    
    def __init__(self):
        self.is_running = False
        self.collectors = {
            'netstat': NetstatCollector() if settings.NETSTAT_ENABLED else None,
            'pcap': PcapCollector() if settings.PCAP_ENABLED else None,
            'log': LogCollector() if settings.LOG_PARSING_ENABLED else None
        }
        self.connection_history = []
        self.stats_history = []
        self.last_stats = None
        
        logger.info("NetworkMonitor initialized")
    
    def is_active(self) -> bool:
        """Check if monitoring is active"""
        return self.is_running
    
    async def start(self):
        """Start network monitoring"""
        self.is_running = True
        logger.info("Network monitoring started")
    
    async def stop(self):
        """Stop network monitoring"""
        self.is_running = False
        logger.info("Network monitoring stopped")
    
    async def collect_data(self) -> List[Dict[str, Any]]:
        """Collect network data from all sources"""
        if not self.is_running:
            return []
        
        try:
            all_data = []
            
            # Collect from netstat
            if self.collectors['netstat']:
                netstat_data = await self.collectors['netstat'].collect()
                all_data.extend(netstat_data)
            
            # Collect system network statistics
            network_stats = await self._collect_network_stats()
            if network_stats:
                all_data.append(network_stats.to_dict())
            
            # Collect active connections
            connections = await self._collect_active_connections()
            all_data.extend([conn.to_dict() for conn in connections])
            
            # Store in history (keep last 1000 entries)
            self.connection_history.extend(all_data)
            self.connection_history = self.connection_history[-1000:]
            
            logger.debug(f"Collected {len(all_data)} network data points")
            return all_data
            
        except Exception as e:
            logger.error(f"Error collecting network data: {e}")
            return []
    
    async def _collect_network_stats(self) -> Optional[NetworkStats]:
        """Collect system network statistics"""
        try:
            stats = psutil.net_io_counters()
            connections = psutil.net_connections()
            
            active_connections = len([c for c in connections if c.status == 'ESTABLISHED'])
            
            network_stats = NetworkStats(
                timestamp=datetime.utcnow().isoformat(),
                total_connections=len(connections),
                active_connections=active_connections,
                bytes_sent_total=stats.bytes_sent,
                bytes_recv_total=stats.bytes_recv,
                packets_sent=stats.packets_sent,
                packets_recv=stats.packets_recv,
                errors_in=stats.errin,
                errors_out=stats.errout,
                drops_in=stats.dropin,
                drops_out=stats.dropout
            )
            
            # Store in history
            self.stats_history.append(network_stats)
            self.stats_history = self.stats_history[-100:]  # Keep last 100
            
            return network_stats
            
        except Exception as e:
            logger.error(f"Error collecting network stats: {e}")
            return None
    
    async def _collect_active_connections(self) -> List[NetworkConnection]:
        """Collect active network connections"""
        connections = []
        
        try:
            net_connections = psutil.net_connections(kind='inet')
            
            for conn in net_connections:
                if conn.status == 'ESTABLISHED':
                    try:
                        # Get process info
                        process_name = None
                        if conn.pid:
                            try:
                                process = psutil.Process(conn.pid)
                                process_name = process.name()
                            except (psutil.NoSuchProcess, psutil.AccessDenied):
                                pass
                        
                        # Extract IP addresses and ports
                        local_ip = conn.laddr.ip if conn.laddr else "0.0.0.0"
                        local_port = conn.laddr.port if conn.laddr else 0
                        remote_ip = conn.raddr.ip if conn.raddr else "0.0.0.0"
                        remote_port = conn.raddr.port if conn.raddr else 0
                        
                        connection = NetworkConnection(
                            timestamp=datetime.utcnow().isoformat(),
                            src_ip=local_ip,
                            dst_ip=remote_ip,
                            src_port=local_port,
                            dst_port=remote_port,
                            protocol=conn.type.name.lower() if conn.type else "tcp",
                            state=conn.status,
                            process_name=process_name
                        )
                        
                        connections.append(connection)
                        
                    except Exception as e:
                        logger.debug(f"Error processing connection: {e}")
                        continue
            
            logger.debug(f"Collected {len(connections)} active connections")
            return connections
            
        except Exception as e:
            logger.error(f"Error collecting active connections: {e}")
            return []
    
    async def get_connection_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get connection history"""
        return self.connection_history[-limit:]
    
    async def get_stats_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get network statistics history"""
        return [stats.__dict__ for stats in self.stats_history[-limit:]]
    
    async def get_current_stats(self) -> Dict[str, Any]:
        """Get current network statistics"""
        try:
            stats = psutil.net_io_counters()
            connections = psutil.net_connections()
            
            # Calculate rates if we have previous stats
            bytes_sent_rate = 0
            bytes_recv_rate = 0
            packets_sent_rate = 0
            packets_recv_rate = 0
            
            if self.last_stats:
                time_diff = (datetime.utcnow() - self.last_stats['timestamp']).total_seconds()
                if time_diff > 0:
                    bytes_sent_rate = (stats.bytes_sent - self.last_stats['bytes_sent']) / time_diff
                    bytes_recv_rate = (stats.bytes_recv - self.last_stats['bytes_recv']) / time_diff
                    packets_sent_rate = (stats.packets_sent - self.last_stats['packets_sent']) / time_diff
                    packets_recv_rate = (stats.packets_recv - self.last_stats['packets_recv']) / time_diff
            
            current_stats = {
                'timestamp': datetime.utcnow(),
                'total_connections': len(connections),
                'active_connections': len([c for c in connections if c.status == 'ESTABLISHED']),
                'bytes_sent': stats.bytes_sent,
                'bytes_recv': stats.bytes_recv,
                'packets_sent': stats.packets_sent,
                'packets_recv': stats.packets_recv,
                'bytes_sent_rate': max(0, bytes_sent_rate),
                'bytes_recv_rate': max(0, bytes_recv_rate),
                'packets_sent_rate': max(0, packets_sent_rate),
                'packets_recv_rate': max(0, packets_recv_rate),
                'errors_in': stats.errin,
                'errors_out': stats.errout
            }
            
            self.last_stats = current_stats.copy()
            return current_stats
            
        except Exception as e:
            logger.error(f"Error getting current stats: {e}")
            return {}
    
    async def get_protocol_distribution(self) -> Dict[str, int]:
        """Get protocol distribution from recent connections"""
        try:
            protocol_counts = {}
            recent_connections = self.connection_history[-200:]  # Last 200 connections
            
            for conn in recent_connections:
                protocol = conn.get('protocol', 'unknown')
                protocol_counts[protocol] = protocol_counts.get(protocol, 0) + 1
            
            return protocol_counts
            
        except Exception as e:
            logger.error(f"Error getting protocol distribution: {e}")
            return {}
    
    async def get_top_talkers(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top talking IP addresses"""
        try:
            ip_stats = {}
            recent_connections = self.connection_history[-500:]  # Last 500 connections
            
            for conn in recent_connections:
                src_ip = conn.get('src_ip', 'unknown')
                dst_ip = conn.get('dst_ip', 'unknown')
                
                # Count source IPs
                if src_ip != 'unknown':
                    if src_ip not in ip_stats:
                        ip_stats[src_ip] = {'connections': 0, 'bytes_sent': 0, 'bytes_recv': 0}
                    ip_stats[src_ip]['connections'] += 1
                    ip_stats[src_ip]['bytes_sent'] += conn.get('bytes_sent', 0)
                
                # Count destination IPs
                if dst_ip != 'unknown':
                    if dst_ip not in ip_stats:
                        ip_stats[dst_ip] = {'connections': 0, 'bytes_sent': 0, 'bytes_recv': 0}
                    ip_stats[dst_ip]['bytes_recv'] += conn.get('bytes_recv', 0)
            
            # Sort by total bytes (sent + received)
            sorted_ips = sorted(
                ip_stats.items(),
                key=lambda x: x[1]['bytes_sent'] + x[1]['bytes_recv'],
                reverse=True
            )
            
            return [
                {
                    'ip': ip,
                    'connections': stats['connections'],
                    'bytes_sent': stats['bytes_sent'],
                    'bytes_recv': stats['bytes_recv'],
                    'total_bytes': stats['bytes_sent'] + stats['bytes_recv']
                }
                for ip, stats in sorted_ips[:limit]
            ]
            
        except Exception as e:
            logger.error(f"Error getting top talkers: {e}")
            return []
    
    def _is_private_ip(self, ip: str) -> bool:
        """Check if IP address is private"""
        try:
            octets = ip.split('.')
            if len(octets) != 4:
                return False
            
            first_octet = int(octets[0])
            second_octet = int(octets[1])
            
            # Private IP ranges
            if first_octet == 10:
                return True
            elif first_octet == 172 and 16 <= second_octet <= 31:
                return True
            elif first_octet == 192 and second_octet == 168:
                return True
            
            return False
            
        except (ValueError, IndexError):
            return False