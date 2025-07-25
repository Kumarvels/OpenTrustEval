"""
Security Monitor for OpenTrustEval
Real-time security monitoring and threat detection
"""

import os
import json
import logging
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import time

class ThreatLevel(Enum):
    """Threat levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AlertType(Enum):
    """Alert types"""
    AUTHENTICATION_FAILURE = "authentication_failure"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    UNUSUAL_ACCESS_PATTERN = "unusual_access_pattern"
    SECURITY_VIOLATION = "security_violation"
    SYSTEM_COMPROMISE = "system_compromise"

@dataclass
class SecurityEvent:
    """Security event information"""
    event_id: str
    timestamp: datetime
    event_type: str
    source_ip: str
    user_id: Optional[str]
    details: Dict[str, Any]
    threat_level: ThreatLevel
    alert_type: Optional[AlertType] = None

@dataclass
class SecurityAlert:
    """Security alert information"""
    alert_id: str
    timestamp: datetime
    alert_type: AlertType
    threat_level: ThreatLevel
    description: str
    source_ip: str
    user_id: Optional[str]
    details: Dict[str, Any]
    resolved: bool = False
    resolved_at: Optional[datetime] = None

class SecurityMonitor:
    """
    Real-time security monitoring and threat detection
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.events: List[SecurityEvent] = []
        self.alerts: List[SecurityAlert] = []
        self.rules: Dict[str, Dict] = {}
        self.thresholds: Dict[str, int] = {}
        self.blocked_ips: Dict[str, datetime] = {}
        self.suspicious_users: Dict[str, List[datetime]] = {}
        
        # Monitoring settings
        self.max_events = 10000
        self.alert_retention_days = 30
        self.ip_block_duration_minutes = 30
        self.max_failed_attempts = 5
        self.suspicious_activity_threshold = 10
        
        # Callbacks for external integrations
        self.alert_callbacks: List[Callable] = []
        self.block_callbacks: List[Callable] = []
        
        # Threading for background monitoring
        self.monitoring_active = False
        self.monitor_thread = None
        
        # Logging
        self.logger = logging.getLogger('SecurityMonitor')
        
        # Load configuration
        self._load_config(config_path)
        self._setup_default_rules()
        
        # Start monitoring
        self.start_monitoring()
    
    def _load_config(self, config_path: Optional[str]):
        """Load security monitoring configuration"""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                
            self.thresholds = config.get('thresholds', self.thresholds)
            self.max_events = config.get('max_events', self.max_events)
            self.alert_retention_days = config.get('alert_retention_days', self.alert_retention_days)
            self.ip_block_duration_minutes = config.get('ip_block_duration_minutes', self.ip_block_duration_minutes)
            self.max_failed_attempts = config.get('max_failed_attempts', self.max_failed_attempts)
            self.suspicious_activity_threshold = config.get('suspicious_activity_threshold', self.suspicious_activity_threshold)
    
    def _setup_default_rules(self):
        """Setup default security rules"""
        self.rules = {
            'authentication_failure': {
                'threshold': 5,
                'time_window_minutes': 10,
                'action': 'block_ip',
                'threat_level': ThreatLevel.MEDIUM
            },
            'rate_limit_exceeded': {
                'threshold': 100,
                'time_window_minutes': 1,
                'action': 'block_ip',
                'threat_level': ThreatLevel.HIGH
            },
            'suspicious_activity': {
                'threshold': 10,
                'time_window_minutes': 5,
                'action': 'alert',
                'threat_level': ThreatLevel.MEDIUM
            },
            'admin_access': {
                'threshold': 1,
                'time_window_minutes': 60,
                'action': 'alert',
                'threat_level': ThreatLevel.HIGH
            },
            'data_access': {
                'threshold': 50,
                'time_window_minutes': 10,
                'action': 'alert',
                'threat_level': ThreatLevel.MEDIUM
            }
        }
    
    def record_event(self, event_type: str, source_ip: str, user_id: Optional[str] = None, 
                    details: Optional[Dict[str, Any]] = None, threat_level: ThreatLevel = ThreatLevel.LOW):
        """Record a security event"""
        event = SecurityEvent(
            event_id=secrets.token_urlsafe(16),
            timestamp=datetime.now(),
            event_type=event_type,
            source_ip=source_ip,
            user_id=user_id,
            details=details or {},
            threat_level=threat_level
        )
        
        self.events.append(event)
        
        # Maintain event list size
        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events:]
        
        # Check for rule violations
        self._check_rules(event)
        
        # Log event
        self.logger.info(f"Security event recorded: {event_type} from {source_ip}")
        
        return event
    
    def _check_rules(self, event: SecurityEvent):
        """Check if event violates any security rules"""
        if event.event_type in self.rules:
            rule = self.rules[event.event_type]
            
            # Count events in time window
            time_window = timedelta(minutes=rule['time_window_minutes'])
            cutoff_time = datetime.now() - time_window
            
            recent_events = [
                e for e in self.events
                if e.event_type == event.event_type and
                e.source_ip == event.source_ip and
                e.timestamp > cutoff_time
            ]
            
            if len(recent_events) >= rule['threshold']:
                self._trigger_rule_violation(event, rule, recent_events)
    
    def _trigger_rule_violation(self, event: SecurityEvent, rule: Dict, recent_events: List[SecurityEvent]):
        """Trigger action for rule violation"""
        action = rule['action']
        threat_level = rule['threat_level']
        
        if action == 'block_ip':
            self.block_ip(event.source_ip, f"Rule violation: {event.event_type}")
        elif action == 'alert':
            self.create_alert(
                AlertType.SECURITY_VIOLATION,
                threat_level,
                f"Rule violation detected: {event.event_type}",
                event.source_ip,
                event.user_id,
                {
                    'rule': event.event_type,
                    'threshold': rule['threshold'],
                    'time_window': rule['time_window_minutes'],
                    'event_count': len(recent_events)
                }
            )
    
    def create_alert(self, alert_type: AlertType, threat_level: ThreatLevel, description: str,
                    source_ip: str, user_id: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        """Create a security alert"""
        alert = SecurityAlert(
            alert_id=secrets.token_urlsafe(16),
            timestamp=datetime.now(),
            alert_type=alert_type,
            threat_level=threat_level,
            description=description,
            source_ip=source_ip,
            user_id=user_id,
            details=details or {}
        )
        
        self.alerts.append(alert)
        
        # Log alert
        self.logger.warning(f"Security alert created: {alert_type.value} - {description}")
        
        # Trigger callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Alert callback failed: {e}")
        
        return alert
    
    def block_ip(self, ip_address: str, reason: str):
        """Block an IP address"""
        block_until = datetime.now() + timedelta(minutes=self.ip_block_duration_minutes)
        self.blocked_ips[ip_address] = block_until
        
        self.logger.warning(f"IP blocked: {ip_address} - {reason}")
        
        # Trigger callbacks
        for callback in self.block_callbacks:
            try:
                callback(ip_address, reason, block_until)
            except Exception as e:
                self.logger.error(f"Block callback failed: {e}")
    
    def is_ip_blocked(self, ip_address: str) -> bool:
        """Check if IP address is blocked"""
        if ip_address not in self.blocked_ips:
            return False
        
        block_until = self.blocked_ips[ip_address]
        if datetime.now() > block_until:
            # Remove expired block
            del self.blocked_ips[ip_address]
            return False
        
        return True
    
    def unblock_ip(self, ip_address: str):
        """Unblock an IP address"""
        if ip_address in self.blocked_ips:
            del self.blocked_ips[ip_address]
            self.logger.info(f"IP unblocked: {ip_address}")
    
    def record_authentication_failure(self, source_ip: str, user_id: Optional[str] = None, 
                                    details: Optional[Dict[str, Any]] = None):
        """Record authentication failure"""
        self.record_event(
            'authentication_failure',
            source_ip,
            user_id,
            details,
            ThreatLevel.MEDIUM
        )
        
        # Check for suspicious user activity
        if user_id:
            self._check_suspicious_user(user_id, source_ip)
    
    def record_authentication_success(self, source_ip: str, user_id: str, 
                                    details: Optional[Dict[str, Any]] = None):
        """Record successful authentication"""
        self.record_event(
            'authentication_success',
            source_ip,
            user_id,
            details,
            ThreatLevel.LOW
        )
    
    def record_rate_limit_exceeded(self, source_ip: str, endpoint: str, 
                                 details: Optional[Dict[str, Any]] = None):
        """Record rate limit exceeded"""
        event_details = details or {}
        event_details['endpoint'] = endpoint
        
        self.record_event(
            'rate_limit_exceeded',
            source_ip,
            None,
            event_details,
            ThreatLevel.HIGH
        )
    
    def record_admin_access(self, source_ip: str, user_id: str, action: str,
                          details: Optional[Dict[str, Any]] = None):
        """Record admin access"""
        event_details = details or {}
        event_details['action'] = action
        
        self.record_event(
            'admin_access',
            source_ip,
            user_id,
            event_details,
            ThreatLevel.HIGH
        )
    
    def record_data_access(self, source_ip: str, user_id: str, data_type: str,
                         details: Optional[Dict[str, Any]] = None):
        """Record data access"""
        event_details = details or {}
        event_details['data_type'] = data_type
        
        self.record_event(
            'data_access',
            source_ip,
            user_id,
            event_details,
            ThreatLevel.MEDIUM
        )
    
    def _check_suspicious_user(self, user_id: str, source_ip: str):
        """Check for suspicious user activity"""
        if user_id not in self.suspicious_users:
            self.suspicious_users[user_id] = []
        
        self.suspicious_users[user_id].append(datetime.now())
        
        # Remove old entries
        cutoff_time = datetime.now() - timedelta(minutes=10)
        self.suspicious_users[user_id] = [
            t for t in self.suspicious_users[user_id] if t > cutoff_time
        ]
        
        # Check threshold
        if len(self.suspicious_users[user_id]) >= self.max_failed_attempts:
            self.create_alert(
                AlertType.SUSPICIOUS_ACTIVITY,
                ThreatLevel.HIGH,
                f"Suspicious activity detected for user {user_id}",
                source_ip,
                user_id,
                {
                    'failed_attempts': len(self.suspicious_users[user_id]),
                    'time_window_minutes': 10
                }
            )
    
    def add_alert_callback(self, callback: Callable[[SecurityAlert], None]):
        """Add callback for security alerts"""
        self.alert_callbacks.append(callback)
    
    def add_block_callback(self, callback: Callable[[str, str, datetime], None]):
        """Add callback for IP blocks"""
        self.block_callbacks.append(callback)
    
    def start_monitoring(self):
        """Start background monitoring"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            self.logger.info("Security monitoring started")
    
    def stop_monitoring(self):
        """Stop background monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join()
        self.logger.info("Security monitoring stopped")
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.monitoring_active:
            try:
                # Clean up expired blocks
                self._cleanup_expired_blocks()
                
                # Clean up old alerts
                self._cleanup_old_alerts()
                
                # Check for unusual patterns
                self._check_unusual_patterns()
                
                # Sleep for monitoring interval
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)
    
    def _cleanup_expired_blocks(self):
        """Clean up expired IP blocks"""
        current_time = datetime.now()
        expired_ips = [
            ip for ip, block_until in self.blocked_ips.items()
            if current_time > block_until
        ]
        
        for ip in expired_ips:
            del self.blocked_ips[ip]
            self.logger.info(f"IP block expired: {ip}")
    
    def _cleanup_old_alerts(self):
        """Clean up old alerts"""
        cutoff_time = datetime.now() - timedelta(days=self.alert_retention_days)
        self.alerts = [alert for alert in self.alerts if alert.timestamp > cutoff_time]
    
    def _check_unusual_patterns(self):
        """Check for unusual access patterns"""
        # Check for rapid authentication attempts from same IP
        recent_events = [
            e for e in self.events
            if e.timestamp > datetime.now() - timedelta(minutes=5)
        ]
        
        ip_counts = {}
        for event in recent_events:
            ip_counts[event.source_ip] = ip_counts.get(event.source_ip, 0) + 1
        
        for ip, count in ip_counts.items():
            if count > self.suspicious_activity_threshold:
                self.create_alert(
                    AlertType.UNUSUAL_ACCESS_PATTERN,
                    ThreatLevel.MEDIUM,
                    f"Unusual activity detected from {ip}",
                    ip,
                    None,
                    {'event_count': count, 'time_window_minutes': 5}
                )
    
    def get_security_summary(self) -> Dict[str, Any]:
        """Get security monitoring summary"""
        current_time = datetime.now()
        
        # Recent events (last 24 hours)
        cutoff_time = current_time - timedelta(hours=24)
        recent_events = [e for e in self.events if e.timestamp > cutoff_time]
        
        # Active alerts
        active_alerts = [a for a in self.alerts if not a.resolved]
        
        # Blocked IPs
        active_blocks = len([ip for ip, block_until in self.blocked_ips.items() if current_time <= block_until])
        
        # Threat level distribution
        threat_levels = {}
        for level in ThreatLevel:
            threat_levels[level.value] = len([e for e in recent_events if e.threat_level == level])
        
        return {
            'timestamp': current_time.isoformat(),
            'total_events_24h': len(recent_events),
            'active_alerts': len(active_alerts),
            'blocked_ips': active_blocks,
            'threat_level_distribution': threat_levels,
            'suspicious_users': len(self.suspicious_users)
        }
    
    def get_recent_alerts(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent alerts"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_alerts = [a for a in self.alerts if a.timestamp > cutoff_time]
        
        return [asdict(alert) for alert in recent_alerts]
    
    def get_blocked_ips(self) -> List[Dict[str, Any]]:
        """Get currently blocked IPs"""
        current_time = datetime.now()
        active_blocks = [
            {
                'ip_address': ip,
                'blocked_until': block_until.isoformat(),
                'remaining_minutes': int((block_until - current_time).total_seconds() / 60)
            }
            for ip, block_until in self.blocked_ips.items()
            if current_time <= block_until
        ]
        
        return active_blocks
    
    def resolve_alert(self, alert_id: str, resolution_notes: Optional[str] = None):
        """Resolve a security alert"""
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.resolved = True
                alert.resolved_at = datetime.now()
                if resolution_notes:
                    alert.details['resolution_notes'] = resolution_notes
                
                self.logger.info(f"Alert resolved: {alert_id}")
                break
    
    def export_events(self, format: str = 'json', output_path: Optional[str] = None) -> str:
        """Export security events"""
        if format.lower() == 'json':
            events_data = [asdict(event) for event in self.events]
            content = json.dumps(events_data, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(content)
            self.logger.info(f"Events exported to {output_path}")
        
        return content
    
    def export_alerts(self, format: str = 'json', output_path: Optional[str] = None) -> str:
        """Export security alerts"""
        if format.lower() == 'json':
            alerts_data = [asdict(alert) for alert in self.alerts]
            content = json.dumps(alerts_data, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(content)
            self.logger.info(f"Alerts exported to {output_path}")
        
        return content 