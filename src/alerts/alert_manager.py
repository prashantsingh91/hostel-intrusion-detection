"""
Alert Manager for handling intrusion alerts with cooldown and queue management
"""
import time
from typing import List, Dict, Any, Optional
from datetime import datetime


class AlertManager:
    """Manages intrusion alerts with cooldown system and queue management"""
    
    def __init__(self, max_alerts: int = 5, cooldown_seconds: int = 60):
        self.active_alerts: List[Dict[str, Any]] = []
        self.alert_history: List[Dict[str, Any]] = []
        self.cooldowns: Dict[str, float] = {}  # bbox_key -> last_alert_time
        self.max_alerts = max_alerts
        self.cooldown_seconds = cooldown_seconds
        
        print(f"🚨 Alert Manager initialized (max: {max_alerts}, cooldown: {cooldown_seconds}s)")
    
    def should_alert(self, bbox_key: str, current_time: float) -> bool:
        """Check if enough time has passed since last alert for this location"""
        if bbox_key not in self.cooldowns:
            return True
        
        time_since_last = current_time - self.cooldowns[bbox_key]
        return time_since_last >= self.cooldown_seconds
    
    def add_alert(self, alert_data: Dict[str, Any]) -> bool:
        """Add new alert to queue"""
        try:
            # Add timestamp if not present
            if 'timestamp' not in alert_data:
                alert_data['timestamp'] = datetime.now().isoformat()
            
            # Add to active alerts
            self.active_alerts.insert(0, alert_data)  # Most recent first
            
            # Limit active alerts
            if len(self.active_alerts) > self.max_alerts:
                # Move oldest to history
                oldest = self.active_alerts.pop()
                self.alert_history.insert(0, oldest)
            
            # Add to history for tracking
            self.alert_history.insert(0, alert_data.copy())
            
            # Limit history size (keep last 100)
            if len(self.alert_history) > 100:
                self.alert_history = self.alert_history[:100]
            
            print(f"🚨 Alert added: {alert_data.get('timestamp', 'unknown')}")
            return True
            
        except Exception as e:
            print(f"❌ Error adding alert: {e}")
            return False
    
    def clear_alerts(self) -> int:
        """Clear all active alerts"""
        count = len(self.active_alerts)
        self.active_alerts = []
        print(f"🧹 Cleared {count} active alerts")
        return count
    
    def get_active_alerts(self) -> Dict[str, Any]:
        """Return active alerts as JSON-ready dict"""
        return {
            'alerts': self.active_alerts,
            'count': len(self.active_alerts),
            'max_alerts': self.max_alerts
        }
    
    def get_alert_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent alert history"""
        return self.alert_history[:limit]
    
    def cleanup_old_cooldowns(self, current_time: float, timeout: float = 300.0):
        """Remove expired cooldowns (older than 5 minutes)"""
        expired_keys = []
        for bbox_key, last_alert_time in self.cooldowns.items():
            if current_time - last_alert_time > timeout:
                expired_keys.append(bbox_key)
        
        for key in expired_keys:
            del self.cooldowns[key]
        
        if expired_keys:
            print(f"🧹 Cleaned up {len(expired_keys)} expired cooldowns")
    
    def update_cooldown(self, bbox_key: str, current_time: float):
        """Update cooldown timestamp for a location"""
        self.cooldowns[bbox_key] = current_time
    
    def get_stats(self) -> Dict[str, Any]:
        """Get alert statistics"""
        return {
            'active_alerts': len(self.active_alerts),
            'total_history': len(self.alert_history),
            'active_cooldowns': len(self.cooldowns),
            'max_alerts': self.max_alerts,
            'cooldown_seconds': self.cooldown_seconds
        }
    
    def get_recent_alerts(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get alerts from the last N hours"""
        try:
            cutoff_time = datetime.now().timestamp() - (hours * 3600)
            recent_alerts = []
            
            for alert in self.alert_history:
                alert_time = datetime.fromisoformat(alert['timestamp']).timestamp()
                if alert_time >= cutoff_time:
                    recent_alerts.append(alert)
            
            return recent_alerts
            
        except Exception as e:
            print(f"❌ Error getting recent alerts: {e}")
            return []
    
    def mark_alert_resolved(self, alert_id: str) -> bool:
        """Mark a specific alert as resolved"""
        try:
            for alert in self.active_alerts:
                if alert.get('id') == alert_id:
                    alert['status'] = 'resolved'
                    alert['resolved_at'] = datetime.now().isoformat()
                    return True
            return False
        except Exception as e:
            print(f"❌ Error marking alert resolved: {e}")
            return False