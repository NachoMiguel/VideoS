"""
Credit Management System for AI Video Slicer
Handles multi-account rotation, usage tracking, and credit protection.
"""
import os
import asyncio
import json
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import aiofiles
from pathlib import Path

from .logger import logger
from .exceptions import *
from .config import settings

class ServiceType(Enum):
    OPENAI = "openai"
    ELEVENLABS = "elevenlabs"
    GOOGLE_SEARCH = "google_search"

@dataclass
class AccountInfo:
    """Information about an API account."""
    service: ServiceType
    api_key: str
    account_id: str
    monthly_limit: int
    current_usage: int = 0
    last_reset: datetime = None
    is_active: bool = True
    failure_count: int = 0
    last_failure: datetime = None

@dataclass
class UsageRecord:
    """Record of API usage for tracking."""
    service: ServiceType
    account_id: str
    operation: str
    cost_estimate: float
    timestamp: datetime
    session_id: str
    success: bool = True
    error_message: str = None

class CreditManager:
    """Manages API credits, account rotation, and usage tracking."""
    
    def __init__(self):
        self.accounts: Dict[ServiceType, List[AccountInfo]] = {}
        self.usage_records: List[UsageRecord] = []
        self.current_session_id = None
        self.session_start_time = None
        self._initialize_accounts()
        
    def _initialize_accounts(self):
        """Initialize accounts from environment variables."""
        # OpenAI Account
        openai_key = os.getenv('OPENAI_API_KEY')
        if openai_key:
            self.accounts[ServiceType.OPENAI] = [
                AccountInfo(
                    service=ServiceType.OPENAI,
                    api_key=openai_key,
                    account_id="openai_main",
                    monthly_limit=10,  # $10 budget
                    last_reset=datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
                )
            ]
        
        # ElevenLabs Accounts (4 accounts)
        elevenlabs_accounts = []
        for i in range(1, 5):
            key_name = f'ELEVENLABS_API_KEY{"" if i == 1 else f"_{i}"}'
            api_key = os.getenv(key_name)
            if api_key:
                elevenlabs_accounts.append(
                    AccountInfo(
                        service=ServiceType.ELEVENLABS,
                        api_key=api_key,
                        account_id=f"elevenlabs_{i}",
                        monthly_limit=30000,  # 30,000 characters per account
                        last_reset=datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
                    )
                )
        
        if elevenlabs_accounts:
            self.accounts[ServiceType.ELEVENLABS] = elevenlabs_accounts
        
        # Google Custom Search Account
        gcs_key = os.getenv('GOOGLE_CUSTOM_SEARCH_API_KEY')
        if gcs_key:
            self.accounts[ServiceType.GOOGLE_SEARCH] = [
                AccountInfo(
                    service=ServiceType.GOOGLE_SEARCH,
                    api_key=gcs_key,
                    account_id="gcs_main",
                    monthly_limit=100,  # 100 queries per day (using monthly for consistency)
                    last_reset=datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                )
            ]
        
        logger.info(f"Initialized credit manager with accounts: "
                   f"OpenAI: {len(self.accounts.get(ServiceType.OPENAI, []))}, "
                   f"ElevenLabs: {len(self.accounts.get(ServiceType.ELEVENLABS, []))}, "
                   f"Google Search: {len(self.accounts.get(ServiceType.GOOGLE_SEARCH, []))}")
    
    def start_session(self, session_id: str):
        """Start a new processing session."""
        self.current_session_id = session_id
        self.session_start_time = datetime.now()
        self.usage_records = []  # Reset usage records for new session
        logger.info(f"Started new credit tracking session: {session_id}")
    
    def get_available_account(self, service: ServiceType) -> Optional[AccountInfo]:
        """Get the next available account for a service."""
        if service not in self.accounts:
            raise ConfigurationError(f"No accounts configured for {service.value}")
        
        accounts = self.accounts[service]
        
        # Check for monthly reset
        self._check_monthly_reset(service)
        
        # Find account with lowest usage that's under 90% limit
        available_accounts = [
            acc for acc in accounts 
            if acc.is_active and (acc.current_usage / acc.monthly_limit) < 0.90
        ]
        
        if not available_accounts:
            # Check if we're at 95% threshold (pause operations)
            all_at_95 = all(
                (acc.current_usage / acc.monthly_limit) >= 0.95 
                for acc in accounts if acc.is_active
            )
            
            if all_at_95:
                raise CreditExhaustionError(
                    f"All {service.value} accounts have reached 95% credit limit. "
                    f"Operations paused to prevent overage charges."
                )
            
            # If between 90-95%, use account with lowest usage
            available_accounts = [acc for acc in accounts if acc.is_active]
        
        if not available_accounts:
            raise CreditExhaustionError(f"No active accounts available for {service.value}")
        
        # Return account with lowest usage percentage
        return min(available_accounts, key=lambda x: x.current_usage / x.monthly_limit)
    
    def _check_monthly_reset(self, service: ServiceType):
        """Check if monthly usage should be reset."""
        now = datetime.now()
        
        for account in self.accounts.get(service, []):
            if service == ServiceType.GOOGLE_SEARCH:
                # Daily reset for Google Search
                if account.last_reset.date() < now.date():
                    account.current_usage = 0
                    account.last_reset = now.replace(hour=0, minute=0, second=0, microsecond=0)
            else:
                # Monthly reset for others
                if account.last_reset.month != now.month or account.last_reset.year != now.year:
                    account.current_usage = 0
                    account.last_reset = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    
    async def record_usage(self, service: ServiceType, account_id: str, operation: str, 
                          cost_estimate: float, success: bool = True, error_message: str = None):
        """Record API usage for tracking."""
        # Update account usage
        account = self._get_account_by_id(service, account_id)
        if account:
            if success:
                account.current_usage += cost_estimate
                account.failure_count = 0  # Reset failure count on success
            else:
                account.failure_count += 1
                account.last_failure = datetime.now()
                
                # Deactivate account if too many failures
                if account.failure_count >= 3:
                    account.is_active = False
                    logger.warning(f"Deactivated account {account_id} due to repeated failures")
        
        # Record usage
        usage_record = UsageRecord(
            service=service,
            account_id=account_id,
            operation=operation,
            cost_estimate=cost_estimate,
            timestamp=datetime.now(),
            session_id=self.current_session_id,
            success=success,
            error_message=error_message
        )
        
        self.usage_records.append(usage_record)
        
        # Check for alerts
        await self._check_usage_alerts(service, account)
        
        logger.info(f"Recorded usage: {service.value} - {operation} - "
                   f"Cost: {cost_estimate} - Success: {success}")
    
    def _get_account_by_id(self, service: ServiceType, account_id: str) -> Optional[AccountInfo]:
        """Get account by ID."""
        for account in self.accounts.get(service, []):
            if account.account_id == account_id:
                return account
        return None
    
    async def _check_usage_alerts(self, service: ServiceType, account: AccountInfo):
        """Check if usage alerts should be triggered."""
        if not account:
            return
        
        usage_percentage = account.current_usage / account.monthly_limit
        
        if usage_percentage >= 0.90:
            alert_message = (
                f"⚠️ {service.value} account {account.account_id} has reached "
                f"{usage_percentage:.1%} of monthly limit"
            )
            logger.warning(alert_message)
            
            # You can extend this to send notifications if needed
    
    def get_session_usage_summary(self) -> Dict[str, Any]:
        """Get usage summary for current session."""
        if not self.usage_records:
            return {"total_cost": 0, "operations": 0, "services": {}}
        
        summary = {
            "session_id": self.current_session_id,
            "session_start": self.session_start_time.isoformat() if self.session_start_time else None,
            "total_cost": 0,
            "operations": len(self.usage_records),
            "services": {}
        }
        
        for service in ServiceType:
            service_records = [r for r in self.usage_records if r.service == service]
            if service_records:
                service_cost = sum(r.cost_estimate for r in service_records if r.success)
                summary["services"][service.value] = {
                    "operations": len(service_records),
                    "cost": service_cost,
                    "success_rate": len([r for r in service_records if r.success]) / len(service_records)
                }
                summary["total_cost"] += service_cost
        
        return summary
    
    def get_account_status(self) -> Dict[str, Any]:
        """Get current status of all accounts."""
        status = {}
        
        for service, accounts in self.accounts.items():
            status[service.value] = []
            for account in accounts:
                usage_percentage = account.current_usage / account.monthly_limit
                status[service.value].append({
                    "account_id": account.account_id,
                    "usage": account.current_usage,
                    "limit": account.monthly_limit,
                    "usage_percentage": usage_percentage,
                    "is_active": account.is_active,
                    "failure_count": account.failure_count,
                    "status": "healthy" if usage_percentage < 0.90 else "warning" if usage_percentage < 0.95 else "critical"
                })
        
        return status
    
    async def save_session_log(self):
        """Save session usage log to file."""
        if not self.current_session_id:
            return
        
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f"credit_usage_{self.current_session_id}.json"
        
        log_data = {
            "session_summary": self.get_session_usage_summary(),
            "account_status": self.get_account_status(),
            "usage_records": [
                {
                    **asdict(record),
                    "timestamp": record.timestamp.isoformat()
                }
                for record in self.usage_records
            ]
        }
        
        async with aiofiles.open(log_file, 'w') as f:
            await f.write(json.dumps(log_data, indent=2))
        
        logger.info(f"Saved credit usage log to {log_file}")

# Global credit manager instance
credit_manager = CreditManager()

class CreditExhaustionError(AIVideoSlicerException):
    """Raised when API credits are exhausted."""
    def __init__(self, message: str):
        super().__init__(message, "CREDIT_EXHAUSTED") 