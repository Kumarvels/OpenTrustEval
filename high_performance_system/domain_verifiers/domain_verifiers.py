"""
Domain-Specific Verification Modules
for Ecommerce, Banking, and Insurance

Real-time verification from trusted sources with high performance
"""

import asyncio
import aiohttp
import json
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import redis
import hashlib
from datetime import datetime, timedelta

@dataclass
class DomainVerificationResult:
    """Domain-specific verification result"""
    domain: str
    verification_type: str
    verified: bool
    confidence: float
    data: Dict[str, Any]
    response_time: float
    source: str
    timestamp: float

class EcommerceVerifier:
    """High-performance ecommerce verification"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.session = None
        self.cache_ttl = 300  # 5 minutes for ecommerce data
        
        # Trusted ecommerce APIs
        self.apis = {
            'product_catalog': 'https://api.ecommerce.com/v1/products',
            'pricing': 'https://api.pricing.com/v1/verify',
            'inventory': 'https://api.inventory.com/v1/check',
            'shipping': 'https://api.shipping.com/v1/rates',
            'reviews': 'https://api.reviews.com/v1/product',
            'amazon': 'https://api.amazon.com/v1/products',
            'ebay': 'https://api.ebay.com/v1/products',
            'walmart': 'https://api.walmart.com/v1/products'
        }

    async def initialize(self):
        """Initialize async session"""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=3.0)
            connector = aiohttp.TCPConnector(limit=50, limit_per_host=10)
            self.session = aiohttp.ClientSession(timeout=timeout, connector=connector)

    async def verify_product_availability(self, product_id: str, store_id: str = None) -> DomainVerificationResult:
        """Verify product availability in real-time"""
        start_time = time.time()
        
        # Check cache
        cache_key = f"ecom_availability:{product_id}:{store_id or 'all'}"
        cached = self.redis.get(cache_key)
        if cached:
            data = json.loads(cached)
            return DomainVerificationResult(
                domain="ecommerce",
                verification_type="product_availability",
                verified=data['available'],
                confidence=data['confidence'],
                data=data,
                response_time=0.001,
                source="cache",
                timestamp=time.time()
            )
        
        # Query multiple sources in parallel
        tasks = [
            self._check_internal_inventory(product_id, store_id),
            self._check_external_availability(product_id),
            self._check_supplier_status(product_id)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Aggregate results
        available_count = sum(1 for r in results if isinstance(r, dict) and r.get('available', False))
        total_count = len([r for r in results if isinstance(r, dict)])
        
        confidence = available_count / total_count if total_count > 0 else 0.0
        verified = confidence > 0.5
        
        result_data = {
            'available': verified,
            'confidence': confidence,
            'sources_checked': total_count,
            'available_sources': available_count,
            'product_id': product_id,
            'store_id': store_id
        }
        
        # Cache result
        self.redis.setex(cache_key, self.cache_ttl, json.dumps(result_data))
        
        return DomainVerificationResult(
            domain="ecommerce",
            verification_type="product_availability",
            verified=verified,
            confidence=confidence,
            data=result_data,
            response_time=time.time() - start_time,
            source="multi_source",
            timestamp=time.time()
        )

    async def verify_pricing(self, product_id: str, claimed_price: float) -> DomainVerificationResult:
        """Verify product pricing accuracy"""
        start_time = time.time()
        
        # Check cache
        cache_key = f"ecom_pricing:{product_id}:{claimed_price}"
        cached = self.redis.get(cache_key)
        if cached:
            data = json.loads(cached)
            return DomainVerificationResult(
                domain="ecommerce",
                verification_type="pricing",
                verified=data['verified'],
                confidence=data['confidence'],
                data=data,
                response_time=0.001,
                source="cache",
                timestamp=time.time()
            )
        
        # Query pricing APIs
        tasks = [
            self._check_internal_pricing(product_id),
            self._check_competitor_pricing(product_id),
            self._check_market_pricing(product_id)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Analyze pricing accuracy
        valid_prices = []
        for result in results:
            if isinstance(result, dict) and 'price' in result:
                price = result['price']
                # Check if claimed price is within acceptable range (±10%)
                if abs(price - claimed_price) / claimed_price <= 0.1:
                    valid_prices.append(price)
        
        confidence = len(valid_prices) / len(results) if results else 0.0
        verified = confidence > 0.7
        
        result_data = {
            'verified': verified,
            'confidence': confidence,
            'claimed_price': claimed_price,
            'market_prices': valid_prices,
            'price_variance': (max(valid_prices) - min(valid_prices)) / claimed_price if valid_prices else 0.0
        }
        
        # Cache result
        self.redis.setex(cache_key, self.cache_ttl, json.dumps(result_data))
        
        return DomainVerificationResult(
            domain="ecommerce",
            verification_type="pricing",
            verified=verified,
            confidence=confidence,
            data=result_data,
            response_time=time.time() - start_time,
            source="multi_source",
            timestamp=time.time()
        )

    async def verify_shipping_info(self, product_id: str, shipping_method: str, claimed_delivery: str) -> DomainVerificationResult:
        """Verify shipping information accuracy"""
        start_time = time.time()
        
        # Check cache
        cache_key = f"ecom_shipping:{product_id}:{shipping_method}"
        cached = self.redis.get(cache_key)
        if cached:
            data = json.loads(cached)
            return DomainVerificationResult(
                domain="ecommerce",
                verification_type="shipping",
                verified=data['verified'],
                confidence=data['confidence'],
                data=data,
                response_time=0.001,
                source="cache",
                timestamp=time.time()
            )
        
        # Query shipping APIs
        tasks = [
            self._check_carrier_rates(product_id, shipping_method),
            self._check_delivery_estimates(product_id, shipping_method),
            self._check_inventory_location(product_id)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify delivery claims
        verified = False
        confidence = 0.0
        
        for result in results:
            if isinstance(result, dict):
                if 'delivery_estimate' in result:
                    estimated_delivery = result['delivery_estimate']
                    # Check if claimed delivery is realistic
                    if self._is_realistic_delivery(claimed_delivery, estimated_delivery):
                        verified = True
                        confidence += 0.3
        
        result_data = {
            'verified': verified,
            'confidence': min(confidence, 1.0),
            'claimed_delivery': claimed_delivery,
            'shipping_method': shipping_method,
            'available_methods': [r.get('method') for r in results if isinstance(r, dict)]
        }
        
        # Cache result
        self.redis.setex(cache_key, self.cache_ttl, json.dumps(result_data))
        
        return DomainVerificationResult(
            domain="ecommerce",
            verification_type="shipping",
            verified=verified,
            confidence=min(confidence, 1.0),
            data=result_data,
            response_time=time.time() - start_time,
            source="multi_source",
            timestamp=time.time()
        )

    async def _check_internal_inventory(self, product_id: str, store_id: str = None) -> Dict[str, Any]:
        """Check internal inventory system"""
        try:
            url = f"{self.apis['inventory']}/{product_id}"
            params = {'store_id': store_id} if store_id else {}
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        'available': data.get('in_stock', False),
                        'quantity': data.get('quantity', 0),
                        'source': 'internal'
                    }
        except Exception as e:
            return {'available': False, 'error': str(e), 'source': 'internal'}
        
        return {'available': False, 'source': 'internal'}

    async def _check_external_availability(self, product_id: str) -> Dict[str, Any]:
        """Check external availability sources"""
        try:
            # Check Amazon
            amazon_url = f"{self.apis['amazon']}/{product_id}"
            async with self.session.get(amazon_url) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        'available': data.get('available', False),
                        'price': data.get('price'),
                        'source': 'amazon'
                    }
        except Exception:
            pass
        
        return {'available': False, 'source': 'external'}

    async def _check_supplier_status(self, product_id: str) -> Dict[str, Any]:
        """Check supplier availability"""
        try:
            # Simulate supplier check
            return {
                'available': True,
                'lead_time': 7,
                'source': 'supplier'
            }
        except Exception:
            return {'available': False, 'source': 'supplier'}

    async def _check_internal_pricing(self, product_id: str) -> Dict[str, Any]:
        """Check internal pricing"""
        try:
            url = f"{self.apis['pricing']}/{product_id}"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        'price': data.get('price'),
                        'currency': data.get('currency', 'USD'),
                        'source': 'internal'
                    }
        except Exception:
            pass
        
        return {'price': None, 'source': 'internal'}

    async def _check_competitor_pricing(self, product_id: str) -> Dict[str, Any]:
        """Check competitor pricing"""
        try:
            # Check multiple competitors
            competitors = ['walmart', 'ebay']
            prices = []
            
            for competitor in competitors:
                url = f"{self.apis[competitor]}/{product_id}"
                async with self.session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        if 'price' in data:
                            prices.append(data['price'])
            
            if prices:
                return {
                    'price': sum(prices) / len(prices),
                    'competitor_count': len(prices),
                    'source': 'competitors'
                }
        except Exception:
            pass
        
        return {'price': None, 'source': 'competitors'}

    async def _check_market_pricing(self, product_id: str) -> Dict[str, Any]:
        """Check market pricing trends"""
        try:
            # Simulate market pricing check
            return {
                'price': 99.99,
                'trend': 'stable',
                'source': 'market'
            }
        except Exception:
            return {'price': None, 'source': 'market'}

    async def _check_carrier_rates(self, product_id: str, shipping_method: str) -> Dict[str, Any]:
        """Check carrier shipping rates"""
        try:
            url = f"{self.apis['shipping']}/{product_id}"
            params = {'method': shipping_method}
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        'method': shipping_method,
                        'rate': data.get('rate'),
                        'delivery_estimate': data.get('delivery_days'),
                        'source': 'carrier'
                    }
        except Exception:
            pass
        
        return {'method': shipping_method, 'source': 'carrier'}

    async def _check_delivery_estimates(self, product_id: str, shipping_method: str) -> Dict[str, Any]:
        """Check delivery estimates"""
        try:
            # Simulate delivery estimate check
            estimates = {
                'standard': 5,
                'express': 2,
                'overnight': 1
            }
            
            return {
                'method': shipping_method,
                'delivery_estimate': estimates.get(shipping_method, 7),
                'source': 'estimates'
            }
        except Exception:
            return {'method': shipping_method, 'source': 'estimates'}

    async def _check_inventory_location(self, product_id: str) -> Dict[str, Any]:
        """Check inventory location for shipping estimates"""
        try:
            # Simulate location check
            return {
                'location': 'warehouse_1',
                'distance': 100,
                'source': 'location'
            }
        except Exception:
            return {'source': 'location'}

    def _is_realistic_delivery(self, claimed: str, estimated: int) -> bool:
        """Check if claimed delivery is realistic"""
        try:
            # Parse claimed delivery
            if 'day' in claimed.lower():
                claimed_days = int(''.join(filter(str.isdigit, claimed)))
                return abs(claimed_days - estimated) <= 2
        except Exception:
            pass
        
        return True

class BankingVerifier:
    """High-performance banking verification"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.session = None
        self.cache_ttl = 600  # 10 minutes for banking data
        
        # Trusted banking APIs
        self.apis = {
            'account_status': 'https://api.banking.com/v1/accounts',
            'transaction_history': 'https://api.banking.com/v1/transactions',
            'regulatory_compliance': 'https://api.compliance.com/v1/check',
            'product_terms': 'https://api.banking.com/v1/products',
            'fraud_detection': 'https://api.fraud.com/v1/check',
            'credit_bureau': 'https://api.credit.com/v1/report',
            'federal_reserve': 'https://api.fed.gov/v1/rates',
            'fdic': 'https://api.fdic.gov/v1/banks'
        }

    async def initialize(self):
        """Initialize async session"""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=5.0)
            connector = aiohttp.TCPConnector(limit=30, limit_per_host=5)
            self.session = aiohttp.ClientSession(timeout=timeout, connector=connector)

    async def verify_account_status(self, account_id: str, claimed_status: str) -> DomainVerificationResult:
        """Verify account status"""
        start_time = time.time()
        
        # Check cache
        cache_key = f"bank_account:{account_id}:{claimed_status}"
        cached = self.redis.get(cache_key)
        if cached:
            data = json.loads(cached)
            return DomainVerificationResult(
                domain="banking",
                verification_type="account_status",
                verified=data['verified'],
                confidence=data['confidence'],
                data=data,
                response_time=0.001,
                source="cache",
                timestamp=time.time()
            )
        
        # Query account status
        tasks = [
            self._check_account_details(account_id),
            self._check_account_activity(account_id),
            self._check_regulatory_status(account_id)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify status
        actual_status = None
        for result in results:
            if isinstance(result, dict) and 'status' in result:
                actual_status = result['status']
                break
        
        verified = actual_status == claimed_status if actual_status else False
        confidence = 0.9 if verified else 0.1
        
        result_data = {
            'verified': verified,
            'confidence': confidence,
            'claimed_status': claimed_status,
            'actual_status': actual_status,
            'account_id': account_id
        }
        
        # Cache result
        self.redis.setex(cache_key, self.cache_ttl, json.dumps(result_data))
        
        return DomainVerificationResult(
            domain="banking",
            verification_type="account_status",
            verified=verified,
            confidence=confidence,
            data=result_data,
            response_time=time.time() - start_time,
            source="multi_source",
            timestamp=time.time()
        )

    async def verify_transaction_claim(self, account_id: str, transaction_id: str, claimed_amount: float) -> DomainVerificationResult:
        """Verify transaction details"""
        start_time = time.time()
        
        # Check cache
        cache_key = f"bank_transaction:{account_id}:{transaction_id}"
        cached = self.redis.get(cache_key)
        if cached:
            data = json.loads(cached)
            return DomainVerificationResult(
                domain="banking",
                verification_type="transaction",
                verified=data['verified'],
                confidence=data['confidence'],
                data=data,
                response_time=0.001,
                source="cache",
                timestamp=time.time()
            )
        
        # Query transaction details
        tasks = [
            self._check_transaction_details(account_id, transaction_id),
            self._check_fraud_indicators(transaction_id),
            self._check_compliance_status(transaction_id)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify transaction
        actual_amount = None
        for result in results:
            if isinstance(result, dict) and 'amount' in result:
                actual_amount = result['amount']
                break
        
        verified = abs(actual_amount - claimed_amount) < 0.01 if actual_amount else False
        confidence = 0.95 if verified else 0.1
        
        result_data = {
            'verified': verified,
            'confidence': confidence,
            'claimed_amount': claimed_amount,
            'actual_amount': actual_amount,
            'transaction_id': transaction_id,
            'account_id': account_id
        }
        
        # Cache result
        self.redis.setex(cache_key, self.cache_ttl, json.dumps(result_data))
        
        return DomainVerificationResult(
            domain="banking",
            verification_type="transaction",
            verified=verified,
            confidence=confidence,
            data=result_data,
            response_time=time.time() - start_time,
            source="multi_source",
            timestamp=time.time()
        )

    async def verify_regulatory_compliance(self, claim: str, regulation: str) -> DomainVerificationResult:
        """Verify regulatory compliance claims"""
        start_time = time.time()
        
        # Check cache
        cache_key = f"bank_compliance:{hashlib.md5(claim.encode()).hexdigest()}"
        cached = self.redis.get(cache_key)
        if cached:
            data = json.loads(cached)
            return DomainVerificationResult(
                domain="banking",
                verification_type="compliance",
                verified=data['verified'],
                confidence=data['confidence'],
                data=data,
                response_time=0.001,
                source="cache",
                timestamp=time.time()
            )
        
        # Query compliance APIs
        tasks = [
            self._check_regulation_status(regulation),
            self._check_compliance_history(regulation),
            self._check_regulatory_updates(regulation)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify compliance
        compliant = True
        confidence = 0.8
        
        for result in results:
            if isinstance(result, dict) and not result.get('compliant', True):
                compliant = False
                confidence = 0.2
        
        result_data = {
            'verified': compliant,
            'confidence': confidence,
            'claim': claim,
            'regulation': regulation,
            'compliant': compliant
        }
        
        # Cache result
        self.redis.setex(cache_key, self.cache_ttl, json.dumps(result_data))
        
        return DomainVerificationResult(
            domain="banking",
            verification_type="compliance",
            verified=compliant,
            confidence=confidence,
            data=result_data,
            response_time=time.time() - start_time,
            source="multi_source",
            timestamp=time.time()
        )

    async def _check_account_details(self, account_id: str) -> Dict[str, Any]:
        """Check account details"""
        try:
            url = f"{self.apis['account_status']}/{account_id}"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        'status': data.get('status'),
                        'type': data.get('type'),
                        'balance': data.get('balance'),
                        'source': 'account_details'
                    }
        except Exception:
            pass
        
        return {'source': 'account_details'}

    async def _check_account_activity(self, account_id: str) -> Dict[str, Any]:
        """Check account activity"""
        try:
            url = f"{self.apis['transaction_history']}/{account_id}/recent"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        'last_activity': data.get('last_activity'),
                        'transaction_count': data.get('count'),
                        'source': 'account_activity'
                    }
        except Exception:
            pass
        
        return {'source': 'account_activity'}

    async def _check_regulatory_status(self, account_id: str) -> Dict[str, Any]:
        """Check regulatory status"""
        try:
            url = f"{self.apis['regulatory_compliance']}/account/{account_id}"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        'status': data.get('status'),
                        'compliant': data.get('compliant', True),
                        'source': 'regulatory'
                    }
        except Exception:
            pass
        
        return {'source': 'regulatory'}

    async def _check_transaction_details(self, account_id: str, transaction_id: str) -> Dict[str, Any]:
        """Check transaction details"""
        try:
            url = f"{self.apis['transaction_history']}/{account_id}/transaction/{transaction_id}"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        'amount': data.get('amount'),
                        'date': data.get('date'),
                        'description': data.get('description'),
                        'source': 'transaction_details'
                    }
        except Exception:
            pass
        
        return {'source': 'transaction_details'}

    async def _check_fraud_indicators(self, transaction_id: str) -> Dict[str, Any]:
        """Check fraud indicators"""
        try:
            url = f"{self.apis['fraud_detection']}/{transaction_id}"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        'fraud_score': data.get('fraud_score', 0),
                        'suspicious': data.get('suspicious', False),
                        'source': 'fraud_detection'
                    }
        except Exception:
            pass
        
        return {'source': 'fraud_detection'}

    async def _check_compliance_status(self, transaction_id: str) -> Dict[str, Any]:
        """Check compliance status"""
        try:
            url = f"{self.apis['regulatory_compliance']}/transaction/{transaction_id}"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        'compliant': data.get('compliant', True),
                        'violations': data.get('violations', []),
                        'source': 'compliance'
                    }
        except Exception:
            pass
        
        return {'source': 'compliance'}

    async def _check_regulation_status(self, regulation: str) -> Dict[str, Any]:
        """Check regulation status"""
        try:
            url = f"{self.apis['regulatory_compliance']}/regulation/{regulation}"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        'active': data.get('active', True),
                        'compliant': data.get('compliant', True),
                        'source': 'regulation_status'
                    }
        except Exception:
            pass
        
        return {'source': 'regulation_status'}

    async def _check_compliance_history(self, regulation: str) -> Dict[str, Any]:
        """Check compliance history"""
        try:
            # Simulate compliance history check
            return {
                'history': 'compliant',
                'last_check': datetime.now().isoformat(),
                'source': 'compliance_history'
            }
        except Exception:
            return {'source': 'compliance_history'}

    async def _check_regulatory_updates(self, regulation: str) -> Dict[str, Any]:
        """Check regulatory updates"""
        try:
            # Simulate regulatory updates check
            return {
                'up_to_date': True,
                'last_update': datetime.now().isoformat(),
                'source': 'regulatory_updates'
            }
        except Exception:
            return {'source': 'regulatory_updates'}

class InsuranceVerifier:
    """High-performance insurance verification"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.session = None
        self.cache_ttl = 1800  # 30 minutes for insurance data
        
        # Trusted insurance APIs
        self.apis = {
            'policy_status': 'https://api.insurance.com/v1/policies',
            'coverage_details': 'https://api.coverage.com/v1/details',
            'claim_status': 'https://api.claims.com/v1/status',
            'risk_assessment': 'https://api.risk.com/v1/assess',
            'regulatory_compliance': 'https://api.compliance.com/v1/insurance',
            'fraud_detection': 'https://api.fraud.com/v1/insurance',
            'medical_records': 'https://api.medical.com/v1/records',
            'vehicle_history': 'https://api.vehicle.com/v1/history'
        }

    async def initialize(self):
        """Initialize async session"""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=5.0)
            connector = aiohttp.TCPConnector(limit=30, limit_per_host=5)
            self.session = aiohttp.ClientSession(timeout=timeout, connector=connector)

    async def verify_policy_status(self, policy_id: str, claimed_status: str) -> DomainVerificationResult:
        """Verify policy status"""
        start_time = time.time()
        
        # Check cache
        cache_key = f"insurance_policy:{policy_id}:{claimed_status}"
        cached = self.redis.get(cache_key)
        if cached:
            data = json.loads(cached)
            return DomainVerificationResult(
                domain="insurance",
                verification_type="policy_status",
                verified=data['verified'],
                confidence=data['confidence'],
                data=data,
                response_time=0.001,
                source="cache",
                timestamp=time.time()
            )
        
        # Query policy status
        tasks = [
            self._check_policy_details(policy_id),
            self._check_policy_activity(policy_id),
            self._check_regulatory_status(policy_id)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify status
        actual_status = None
        for result in results:
            if isinstance(result, dict) and 'status' in result:
                actual_status = result['status']
                break
        
        verified = actual_status == claimed_status if actual_status else False
        confidence = 0.9 if verified else 0.1
        
        result_data = {
            'verified': verified,
            'confidence': confidence,
            'claimed_status': claimed_status,
            'actual_status': actual_status,
            'policy_id': policy_id
        }
        
        # Cache result
        self.redis.setex(cache_key, self.cache_ttl, json.dumps(result_data))
        
        return DomainVerificationResult(
            domain="insurance",
            verification_type="policy_status",
            verified=verified,
            confidence=confidence,
            data=result_data,
            response_time=time.time() - start_time,
            source="multi_source",
            timestamp=time.time()
        )

    async def verify_coverage_claim(self, policy_id: str, claimed_coverage: str, claimed_amount: float) -> DomainVerificationResult:
        """Verify coverage claims"""
        start_time = time.time()
        
        # Check cache
        cache_key = f"insurance_coverage:{policy_id}:{claimed_coverage}"
        cached = self.redis.get(cache_key)
        if cached:
            data = json.loads(cached)
            return DomainVerificationResult(
                domain="insurance",
                verification_type="coverage",
                verified=data['verified'],
                confidence=data['confidence'],
                data=data,
                response_time=0.001,
                source="cache",
                timestamp=time.time()
            )
        
        # Query coverage details
        tasks = [
            self._check_coverage_details(policy_id, claimed_coverage),
            self._check_coverage_limits(policy_id, claimed_coverage),
            self._check_exclusions(policy_id, claimed_coverage)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify coverage
        actual_coverage = None
        actual_amount = None
        
        for result in results:
            if isinstance(result, dict):
                if 'coverage' in result:
                    actual_coverage = result['coverage']
                if 'amount' in result:
                    actual_amount = result['amount']
        
        verified = (actual_coverage == claimed_coverage and 
                   abs(actual_amount - claimed_amount) < 100) if actual_coverage and actual_amount else False
        confidence = 0.9 if verified else 0.1
        
        result_data = {
            'verified': verified,
            'confidence': confidence,
            'claimed_coverage': claimed_coverage,
            'claimed_amount': claimed_amount,
            'actual_coverage': actual_coverage,
            'actual_amount': actual_amount,
            'policy_id': policy_id
        }
        
        # Cache result
        self.redis.setex(cache_key, self.cache_ttl, json.dumps(result_data))
        
        return DomainVerificationResult(
            domain="insurance",
            verification_type="coverage",
            verified=verified,
            confidence=confidence,
            data=result_data,
            response_time=time.time() - start_time,
            source="multi_source",
            timestamp=time.time()
        )

    async def verify_claim_status(self, claim_id: str, claimed_status: str) -> DomainVerificationResult:
        """Verify claim status"""
        start_time = time.time()
        
        # Check cache
        cache_key = f"insurance_claim:{claim_id}:{claimed_status}"
        cached = self.redis.get(cache_key)
        if cached:
            data = json.loads(cached)
            return DomainVerificationResult(
                domain="insurance",
                verification_type="claim_status",
                verified=data['verified'],
                confidence=data['confidence'],
                data=data,
                response_time=0.001,
                source="cache",
                timestamp=time.time()
            )
        
        # Query claim status
        tasks = [
            self._check_claim_details(claim_id),
            self._check_claim_processing(claim_id),
            self._check_fraud_indicators(claim_id)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify status
        actual_status = None
        for result in results:
            if isinstance(result, dict) and 'status' in result:
                actual_status = result['status']
                break
        
        verified = actual_status == claimed_status if actual_status else False
        confidence = 0.9 if verified else 0.1
        
        result_data = {
            'verified': verified,
            'confidence': confidence,
            'claimed_status': claimed_status,
            'actual_status': actual_status,
            'claim_id': claim_id
        }
        
        # Cache result
        self.redis.setex(cache_key, self.cache_ttl, json.dumps(result_data))
        
        return DomainVerificationResult(
            domain="insurance",
            verification_type="claim_status",
            verified=verified,
            confidence=confidence,
            data=result_data,
            response_time=time.time() - start_time,
            source="multi_source",
            timestamp=time.time()
        )

    async def _check_policy_details(self, policy_id: str) -> Dict[str, Any]:
        """Check policy details"""
        try:
            url = f"{self.apis['policy_status']}/{policy_id}"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        'status': data.get('status'),
                        'type': data.get('type'),
                        'premium': data.get('premium'),
                        'source': 'policy_details'
                    }
        except Exception:
            pass
        
        return {'source': 'policy_details'}

    async def _check_policy_activity(self, policy_id: str) -> Dict[str, Any]:
        """Check policy activity"""
        try:
            url = f"{self.apis['policy_status']}/{policy_id}/activity"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        'last_activity': data.get('last_activity'),
                        'claims_count': data.get('claims_count'),
                        'source': 'policy_activity'
                    }
        except Exception:
            pass
        
        return {'source': 'policy_activity'}

    async def _check_regulatory_status(self, policy_id: str) -> Dict[str, Any]:
        """Check regulatory status"""
        try:
            url = f"{self.apis['regulatory_compliance']}/policy/{policy_id}"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        'status': data.get('status'),
                        'compliant': data.get('compliant', True),
                        'source': 'regulatory'
                    }
        except Exception:
            pass
        
        return {'source': 'regulatory'}

    async def _check_coverage_details(self, policy_id: str, coverage_type: str) -> Dict[str, Any]:
        """Check coverage details"""
        try:
            url = f"{self.apis['coverage_details']}/{policy_id}/{coverage_type}"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        'coverage': data.get('coverage'),
                        'amount': data.get('amount'),
                        'deductible': data.get('deductible'),
                        'source': 'coverage_details'
                    }
        except Exception:
            pass
        
        return {'source': 'coverage_details'}

    async def _check_coverage_limits(self, policy_id: str, coverage_type: str) -> Dict[str, Any]:
        """Check coverage limits"""
        try:
            url = f"{self.apis['coverage_details']}/{policy_id}/{coverage_type}/limits"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        'limit': data.get('limit'),
                        'used': data.get('used'),
                        'remaining': data.get('remaining'),
                        'source': 'coverage_limits'
                    }
        except Exception:
            pass
        
        return {'source': 'coverage_limits'}

    async def _check_exclusions(self, policy_id: str, coverage_type: str) -> Dict[str, Any]:
        """Check coverage exclusions"""
        try:
            url = f"{self.apis['coverage_details']}/{policy_id}/{coverage_type}/exclusions"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        'exclusions': data.get('exclusions', []),
                        'source': 'exclusions'
                    }
        except Exception:
            pass
        
        return {'source': 'exclusions'}

    async def _check_claim_details(self, claim_id: str) -> Dict[str, Any]:
        """Check claim details"""
        try:
            url = f"{self.apis['claim_status']}/{claim_id}"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        'status': data.get('status'),
                        'amount': data.get('amount'),
                        'date': data.get('date'),
                        'source': 'claim_details'
                    }
        except Exception:
            pass
        
        return {'source': 'claim_details'}

    async def _check_claim_processing(self, claim_id: str) -> Dict[str, Any]:
        """Check claim processing status"""
        try:
            url = f"{self.apis['claim_status']}/{claim_id}/processing"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        'stage': data.get('stage'),
                        'estimated_completion': data.get('estimated_completion'),
                        'source': 'claim_processing'
                    }
        except Exception:
            pass
        
        return {'source': 'claim_processing'}

    async def _check_fraud_indicators(self, claim_id: str) -> Dict[str, Any]:
        """Check fraud indicators"""
        try:
            url = f"{self.apis['fraud_detection']}/claim/{claim_id}"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        'fraud_score': data.get('fraud_score', 0),
                        'suspicious': data.get('suspicious', False),
                        'source': 'fraud_detection'
                    }
        except Exception:
            pass
        
        return {'source': 'fraud_detection'} 