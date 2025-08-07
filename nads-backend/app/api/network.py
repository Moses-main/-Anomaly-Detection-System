"""
Network API Endpoints
RESTful API for network monitoring and data collection
"""

from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging

from app.services.network_monitor import NetworkMonitor
from app.core.security import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter()

# Global network monitor instance (will be injected)
network_monitor: Optional[NetworkMonitor] = None

def get_network_monitor() -> NetworkMonitor:
    """Dependency to get network monitor instance"""
    global network_monitor
    if not network_monitor:
        network_monitor = NetworkMonitor()
    return network_monitor

@router.get("/status")
async def get_monitoring_status(
    monitor: NetworkMonitor = Depends(get_network_monitor)
):
    """Get network monitoring status"""
    try:
        return {
            "monitoring_active": monitor.is_active(),
            "timestamp": datetime.utcnow().isoformat(),
            "status": "running" if monitor.is_active() else "stopped"
        }
    except Exception as e:
        logger.error(f"Error getting monitoring status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get monitoring status")

@router.post("/start")
async def start_monitoring(
    monitor: NetworkMonitor = Depends(get_network_monitor)
):
    """Start network monitoring"""
    try:
        await monitor.start()
        return {
            "message": "Network monitoring started",
            "status": "running",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error starting monitoring: {e}")
        raise HTTPException(status_code=500, detail="Failed to start monitoring")

@router.post("/stop")
async def stop_monitoring(
    monitor: NetworkMonitor = Depends(get_network_monitor)
):
    """Stop network monitoring"""
    try:
        await monitor.stop()
        return {
            "message": "Network monitoring stopped",
            "status": "stopped",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error stopping monitoring: {e}")
        raise HTTPException(status_code=500, detail="Failed to stop monitoring")

@router.get("/live")
async def get_live_data(
    limit: int = Query(100, ge=1, le=1000),
    monitor: NetworkMonitor = Depends(get_network_monitor)
):
    """Get live network data"""
    try:
        data = await monitor.collect_data()
        return {
            "data": data[:limit],
            "count": len(data),
            "timestamp": datetime.utcnow().isoformat(),
            "monitoring_active": monitor.is_active()
        }
    except Exception as e:
        logger.error(f"Error getting live data: {e}")
        raise HTTPException(status_code=500, detail="Failed to get live data")

@router.get("/connections")
async def get_connections(
    limit: int = Query(100, ge=1, le=1000),
    monitor: NetworkMonitor = Depends(get_network_monitor)
):
    """Get active network connections"""
    try:
        connections = await monitor.get_connection_history(limit=limit)
        return {
            "connections": connections,
            "count": len(connections),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting connections: {e}")
        raise HTTPException(status_code=500, detail="Failed to get connections")

@router.get("/stats")
async def get_network_stats(
    monitor: NetworkMonitor = Depends(get_network_monitor)
):
    """Get current network statistics"""
    try:
        stats = await monitor.get_current_stats()
        return {
            "stats": stats,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting network stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get network stats")

@router.get("/stats/history")
async def get_stats_history(
    limit: int = Query(50, ge=1, le=200),
    monitor: NetworkMonitor = Depends(get_network_monitor)
):
    """Get network statistics history"""
    try:
        history = await monitor.get_stats_history(limit=limit)
        return {
            "history": history,
            "count": len(history),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting stats history: {e}")
        raise HTTPException(status_code=500, detail="Failed to get stats history")

@router.get("/protocols")
async def get_protocol_distribution(
    monitor: NetworkMonitor = Depends(get_network_monitor)
):
    """Get protocol distribution"""
    try:
        distribution = await monitor.get_protocol_distribution()
        
        # Convert to percentage
        total = sum(distribution.values())
        if total > 0:
            percentages = {protocol: (count/total)*100 for protocol, count in distribution.items()}
        else:
            percentages = {}
        
        return {
            "distribution": distribution,
            "percentages": percentages,
            "total_connections": total,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting protocol distribution: {e}")
        raise HTTPException(status_code=500, detail="Failed to get protocol distribution")

@router.get("/top-talkers")
async def get_top_talkers(
    limit: int = Query(10, ge=1, le=50),
    monitor: NetworkMonitor = Depends(get_network_monitor)
):
    """Get top talking IP addresses"""
    try:
        talkers = await monitor.get_top_talkers(limit=limit)
        return {
            "top_talkers": talkers,
            "count": len(talkers),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting top talkers: {e}")
        raise HTTPException(status_code=500, detail="Failed to get top talkers")

@router.get("/dashboard")
async def get_dashboard_data(
    monitor: NetworkMonitor = Depends(get_network_monitor)
):
    """Get comprehensive dashboard data"""
    try:
        # Collect all dashboard data in parallel
        import asyncio
        
        tasks = [
            monitor.get_current_stats(),
            monitor.get_connection_history(limit=20),
            monitor.get_protocol_distribution(),
            monitor.get_top_talkers(limit=5)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        current_stats = results[0] if not isinstance(results[0], Exception) else {}
        connections = results[1] if not isinstance(results[1], Exception) else []
        protocols = results[2] if not isinstance(results[2], Exception) else {}
        top_talkers = results[3] if not isinstance(results[3], Exception) else []
        
        # Calculate protocol percentages
        total_protocols = sum(protocols.values())
        protocol_percentages = {}
        if total_protocols > 0:
            protocol_percentages = {
                protocol: round((count/total_protocols)*100, 1) 
                for protocol, count in protocols.items()
            }
        
        return {
            "current_stats": current_stats,
            "recent_connections": connections,
            "protocol_distribution": {
                "counts": protocols,
                "percentages": protocol_percentages
            },
            "top_talkers": top_talkers,
            "monitoring_active": monitor.is_active(),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting dashboard data: {e}")
        raise HTTPException(status_code=500, detail="Failed to get dashboard data")

@router.get("/traffic-analysis")
async def get_traffic_analysis(
    time_range: str = Query("1h", regex="^(5m|15m|1h|6h|24h)$"),
    monitor: NetworkMonitor = Depends(get_network_monitor)
):
    """Get traffic analysis for specified time range"""
    try:
        # Convert time range to minutes
        time_mapping = {"5m": 5, "15m": 15, "1h": 60, "6h": 360, "24h": 1440}
        minutes = time_mapping.get(time_range, 60)
        
        # Get recent connections
        connections = await monitor.get_connection_history(limit=1000)
        
        # Filter by time range
        cutoff_time = datetime.utcnow() - timedelta(minutes=minutes)
        recent_connections = [
            conn for conn in connections 
            if datetime.fromisoformat(conn.get('timestamp', '1970-01-01')) > cutoff_time
        ]
        
        # Analyze traffic patterns
        analysis = {
            "time_range": time_range,
            "total_connections": len(recent_connections),
            "unique_source_ips": len(set(conn.get('src_ip') for conn in recent_connections)),
            "unique_dest_ips": len(set(conn.get('dst_ip') for conn in recent_connections)),
            "protocols": {},
            "ports": {},
            "connection_timeline": []
        }
        
        # Protocol analysis
        for conn in recent_connections:
            protocol = conn.get('protocol', 'unknown')
            analysis["protocols"][protocol] = analysis["protocols"].get(protocol, 0) + 1
        
        # Port analysis (top 10 destination ports)
        port_counts = {}
        for conn in recent_connections:
            port = conn.get('dst_port', 0)
            if port > 0:
                port_counts[port] = port_counts.get(port, 0) + 1
        
        analysis["ports"] = dict(sorted(port_counts.items(), key=lambda x: x[1], reverse=True)[:10])
        
        return analysis
        
    except Exception as e:
        logger.error(f"Error getting traffic analysis: {e}")
        raise HTTPException(status_code=500, detail="Failed to get traffic analysis")

@router.post("/collect")
async def trigger_collection(
    background_tasks: BackgroundTasks,
    monitor: NetworkMonitor = Depends(get_network_monitor)
):
    """Manually trigger data collection"""
    try:
        background_tasks.add_task(monitor.collect_data)
        return {
            "message": "Data collection triggered",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error triggering collection: {e}")
        raise HTTPException(status_code=500, detail="Failed to trigger collection")