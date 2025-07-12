"""
Real-Time Verification Orchestrator
High-performance coordination of multiple verification sources

Features:
- Intelligent routing to fastest/most reliable sources
- Real-time load balancing
- Adaptive caching strategies
- Performance monitoring and optimization
- Failover mechanisms
"""

import asyncio
import time
import json
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import redis
import numpy as np
from collections import defaultdict, deque
import logging
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VerificationPriority(Enum):
    """Verification priority levels"""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4

class VerificationType(Enum):
    """Types of verification"""
    FACT_CHECK = "fact_check"
    DOMAIN_SPECIFIC = "domain_specific"
    REAL_TIME = "real_time"
    BATCH = "batch"
    CACHED = "cached"

@dataclass
class VerificationRequest:
    """Verification request with metadata"""
    query: str
    response: str
    domain: str
    priority: VerificationPriority
    verification_type: VerificationType
    context: Dict[str, Any]
    timeout: float = 5.0
    max_sources: int = 5
    timestamp: float = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

@dataclass
class VerificationRoute:
    """Route configuration for verification sources"""
    source_name: str
    priority: int
    weight: float
    timeout: float
    retry_count: int
    enabled: bool
    performance_score: float
    last_response_time: float
    success_rate: float

@dataclass
class OrchestrationResult:
    """Result from verification orchestration"""
    success: bool
    results: List[Any]
    response_time: float
    sources_used: List[str]
    cache_hits: int
    errors: List[str]
    performance_metrics: Dict[str, Any]

class RealTimeVerificationOrchestrator:
    """High-performance real-time verification orchestrator"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.executor = ThreadPoolExecutor(max_workers=50)
        
        # Performance tracking
        self.performance_history = defaultdict(lambda: deque(maxlen=100))
        self.source_health = defaultdict(lambda: {'success': 0, 'failure': 0, 'avg_time': 0.0})
        
        # Routing configuration
        self.routes = self._initialize_routes()
        
        # Cache configuration
        self.cache_ttl = {
            'fact_check': 3600,      # 1 hour
            'domain_specific': 1800,  # 30 minutes
            'real_time': 300,        # 5 minutes
            'batch': 7200           # 2 hours
        }
        
        # Load balancing
        self.current_load = defaultdict(int)
        self.max_concurrent_requests = 100
        
        # Circuit breaker
        self.circuit_breaker = defaultdict(lambda: {'failures': 0, 'last_failure': 0, 'state': 'closed'})
        self.circuit_breaker_threshold = 5
        self.circuit_breaker_timeout = 60  # seconds
        
        # Performance monitoring
        self.metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'avg_response_time': 0.0,
            'cache_hit_rate': 0.0,
            'active_connections': 0
        }

    def _initialize_routes(self) -> Dict[str, VerificationRoute]:
        """Initialize verification routes with performance weights"""
        routes = {
            # Fact checking sources
            'wikipedia': VerificationRoute(
                source_name='wikipedia',
                priority=1,
                weight=0.3,
                timeout=3.0,
                retry_count=2,
                enabled=True,
                performance_score=0.9,
                last_response_time=0.5,
                success_rate=0.95
            ),
            'google_knowledge': VerificationRoute(
                source_name='google_knowledge',
                priority=1,
                weight=0.25,
                timeout=3.0,
                retry_count=2,
                enabled=True,
                performance_score=0.85,
                last_response_time=0.8,
                success_rate=0.90
            ),
            'fact_check_api': VerificationRoute(
                source_name='fact_check_api',
                priority=1,
                weight=0.25,
                timeout=4.0,
                retry_count=1,
                enabled=True,
                performance_score=0.8,
                last_response_time=1.2,
                success_rate=0.85
            ),
            
            # Domain-specific sources
            'ecommerce_verifier': VerificationRoute(
                source_name='ecommerce_verifier',
                priority=2,
                weight=0.4,
                timeout=2.0,
                retry_count=3,
                enabled=True,
                performance_score=0.95,
                last_response_time=0.3,
                success_rate=0.98
            ),
            'banking_verifier': VerificationRoute(
                source_name='banking_verifier',
                priority=2,
                weight=0.4,
                timeout=3.0,
                retry_count=2,
                enabled=True,
                performance_score=0.9,
                last_response_time=0.6,
                success_rate=0.95
            ),
            'insurance_verifier': VerificationRoute(
                source_name='insurance_verifier',
                priority=2,
                weight=0.4,
                timeout=3.0,
                retry_count=2,
                enabled=True,
                performance_score=0.9,
                last_response_time=0.7,
                success_rate=0.93
            ),
            
            # Real-time sources
            'news_api': VerificationRoute(
                source_name='news_api',
                priority=3,
                weight=0.2,
                timeout=2.0,
                retry_count=2,
                enabled=True,
                performance_score=0.7,
                last_response_time=0.9,
                success_rate=0.8
            ),
            'social_media': VerificationRoute(
                source_name='social_media',
                priority=3,
                weight=0.15,
                timeout=1.5,
                retry_count=3,
                enabled=True,
                performance_score=0.6,
                last_response_time=0.4,
                success_rate=0.7
            )
        }
        
        return routes

    async def orchestrate_verification(self, request: VerificationRequest) -> OrchestrationResult:
        """Orchestrate verification across multiple sources"""
        start_time = time.time()
        self.metrics['total_requests'] += 1
        
        try:
            # Check cache first
            cache_result = await self._check_cache(request)
            if cache_result:
                self.metrics['successful_requests'] += 1
                self.metrics['cache_hit_rate'] = self._update_cache_hit_rate(True)
                return OrchestrationResult(
                    success=True,
                    results=[cache_result],
                    response_time=time.time() - start_time,
                    sources_used=['cache'],
                    cache_hits=1,
                    errors=[],
                    performance_metrics=self.metrics
                )
            
            # Select optimal verification sources
            selected_sources = self._select_verification_sources(request)
            
            # Execute verifications in parallel with timeout
            verification_tasks = []
            for source in selected_sources:
                task = self._execute_verification(source, request)
                verification_tasks.append(task)
            
            # Wait for results with timeout
            results = await asyncio.wait_for(
                asyncio.gather(*verification_tasks, return_exceptions=True),
                timeout=request.timeout
            )
            
            # Process results
            successful_results = []
            errors = []
            
            for i, result in enumerate(results):
                source_name = selected_sources[i].source_name
                
                if isinstance(result, Exception):
                    errors.append(f"{source_name}: {str(result)}")
                    self._update_source_health(source_name, False, time.time() - start_time)
                else:
                    successful_results.append(result)
                    self._update_source_health(source_name, True, time.time() - start_time)
            
            # Cache successful results
            if successful_results:
                await self._cache_results(request, successful_results)
            
            success = len(successful_results) > 0
            if success:
                self.metrics['successful_requests'] += 1
            else:
                self.metrics['failed_requests'] += 1
            
            # Update performance metrics
            response_time = time.time() - start_time
            self.metrics['avg_response_time'] = self._update_avg_response_time(response_time)
            
            return OrchestrationResult(
                success=success,
                results=successful_results,
                response_time=response_time,
                sources_used=[r.source_name for r in selected_sources],
                cache_hits=0,
                errors=errors,
                performance_metrics=self.metrics
            )
            
        except asyncio.TimeoutError:
            self.metrics['failed_requests'] += 1
            return OrchestrationResult(
                success=False,
                results=[],
                response_time=time.time() - start_time,
                sources_used=[],
                cache_hits=0,
                errors=['Verification timeout'],
                performance_metrics=self.metrics
            )
        except Exception as e:
            self.metrics['failed_requests'] += 1
            return OrchestrationResult(
                success=False,
                results=[],
                response_time=time.time() - start_time,
                sources_used=[],
                cache_hits=0,
                errors=[str(e)],
                performance_metrics=self.metrics
            )

    async def _check_cache(self, request: VerificationRequest) -> Optional[Dict[str, Any]]:
        """Check cache for existing verification results"""
        cache_key = self._generate_cache_key(request)
        
        try:
            cached_data = self.redis.get(cache_key)
            if cached_data:
                data = json.loads(cached_data)
                
                # Check if cache is still valid
                if time.time() - data.get('timestamp', 0) < self.cache_ttl.get(request.verification_type.value, 3600):
                    return data
        except Exception as e:
            logger.warning(f"Cache check failed: {e}")
        
        return None

    async def _cache_results(self, request: VerificationRequest, results: List[Any]):
        """Cache verification results"""
        cache_key = self._generate_cache_key(request)
        
        try:
            cache_data = {
                'results': results,
                'timestamp': time.time(),
                'sources': [getattr(r, 'source', 'unknown') for r in results],
                'domain': request.domain
            }
            
            ttl = self.cache_ttl.get(request.verification_type.value, 3600)
            self.redis.setex(cache_key, ttl, json.dumps(cache_data))
        except Exception as e:
            logger.warning(f"Caching failed: {e}")

    def _generate_cache_key(self, request: VerificationRequest) -> str:
        """Generate cache key for request"""
        content = f"{request.query}:{request.response}:{request.domain}:{request.verification_type.value}"
        return f"verification:{hashlib.md5(content.encode()).hexdigest()}"

    def _select_verification_sources(self, request: VerificationRequest) -> List[VerificationRoute]:
        """Select optimal verification sources based on request and performance"""
        available_routes = []
        
        # Filter enabled routes
        for route in self.routes.values():
            if not route.enabled:
                continue
            
            # Check circuit breaker
            if self._is_circuit_open(route.source_name):
                continue
            
            # Check load balancing
            if self.current_load[route.source_name] >= self.max_concurrent_requests:
                continue
            
            # Add route based on priority and domain
            if self._is_route_applicable(route, request):
                available_routes.append(route)
        
        # Sort by priority and performance score
        available_routes.sort(key=lambda r: (r.priority, -r.performance_score))
        
        # Select top sources based on request max_sources
        selected = available_routes[:request.max_sources]
        
        # Update load tracking
        for route in selected:
            self.current_load[route.source_name] += 1
        
        return selected

    def _is_route_applicable(self, route: VerificationRoute, request: VerificationRequest) -> bool:
        """Check if route is applicable to the request"""
        # Domain-specific routing
        if request.domain == 'ecommerce' and 'ecommerce' in route.source_name:
            return True
        elif request.domain == 'banking' and 'banking' in route.source_name:
            return True
        elif request.domain == 'insurance' and 'insurance' in route.source_name:
            return True
        
        # General fact checking for all domains
        if route.source_name in ['wikipedia', 'google_knowledge', 'fact_check_api']:
            return True
        
        # Real-time sources for time-sensitive queries
        if request.verification_type == VerificationType.REAL_TIME:
            return route.source_name in ['news_api', 'social_media']
        
        return False

    def _is_circuit_open(self, source_name: str) -> bool:
        """Check if circuit breaker is open for source"""
        circuit = self.circuit_breaker[source_name]
        
        if circuit['state'] == 'open':
            # Check if timeout has passed
            if time.time() - circuit['last_failure'] > self.circuit_breaker_timeout:
                circuit['state'] = 'half_open'
                return False
            return True
        
        return False

    async def _execute_verification(self, route: VerificationRoute, request: VerificationRequest):
        """Execute verification for a specific route"""
        try:
            # Simulate verification execution
            # In production, this would call the actual verification service
            
            if route.source_name == 'wikipedia':
                return await self._verify_wikipedia(request)
            elif route.source_name == 'google_knowledge':
                return await self._verify_google_knowledge(request)
            elif route.source_name == 'fact_check_api':
                return await self._verify_fact_check(request)
            elif route.source_name == 'ecommerce_verifier':
                return await self._verify_ecommerce(request)
            elif route.source_name == 'banking_verifier':
                return await self._verify_banking(request)
            elif route.source_name == 'insurance_verifier':
                return await self._verify_insurance(request)
            else:
                # Generic verification
                return await self._verify_generic(route, request)
                
        except Exception as e:
            # Update circuit breaker
            self._update_circuit_breaker(route.source_name, False)
            raise e
        finally:
            # Update load tracking
            self.current_load[route.source_name] = max(0, self.current_load[route.source_name] - 1)

    def _update_source_health(self, source_name: str, success: bool, response_time: float):
        """Update source health metrics"""
        health = self.source_health[source_name]
        
        if success:
            health['success'] += 1
        else:
            health['failure'] += 1
        
        # Update average response time
        total_requests = health['success'] + health['failure']
        health['avg_time'] = (health['avg_time'] * (total_requests - 1) + response_time) / total_requests
        
        # Update route performance score
        if source_name in self.routes:
            success_rate = health['success'] / total_requests if total_requests > 0 else 0.0
            self.routes[source_name].success_rate = success_rate
            self.routes[source_name].last_response_time = response_time
            self.routes[source_name].performance_score = success_rate * (1.0 / (1.0 + response_time))

    def _update_circuit_breaker(self, source_name: str, success: bool):
        """Update circuit breaker state"""
        circuit = self.circuit_breaker[source_name]
        
        if success:
            circuit['failures'] = 0
            circuit['state'] = 'closed'
        else:
            circuit['failures'] += 1
            circuit['last_failure'] = time.time()
            
            if circuit['failures'] >= self.circuit_breaker_threshold:
                circuit['state'] = 'open'

    def _update_avg_response_time(self, response_time: float) -> float:
        """Update average response time"""
        alpha = 0.1  # Exponential moving average
        self.metrics['avg_response_time'] = (
            alpha * response_time + 
            (1 - alpha) * self.metrics['avg_response_time']
        )
        return self.metrics['avg_response_time']

    def _update_cache_hit_rate(self, cache_hit: bool) -> float:
        """Update cache hit rate"""
        total_requests = self.metrics['total_requests']
        if total_requests == 0:
            return 0.0
        
        # Simple moving average over last 100 requests
        cache_hits = int(self.metrics['cache_hit_rate'] * total_requests)
        if cache_hit:
            cache_hits += 1
        
        return cache_hits / total_requests

    # Placeholder verification methods
    async def _verify_wikipedia(self, request: VerificationRequest):
        """Verify using Wikipedia"""
        await asyncio.sleep(0.5)  # Simulate API call
        return {
            'source': 'wikipedia',
            'verified': True,
            'confidence': 0.8,
            'data': {'extract': 'Sample Wikipedia data'}
        }

    async def _verify_google_knowledge(self, request: VerificationRequest):
        """Verify using Google Knowledge Graph"""
        await asyncio.sleep(0.8)  # Simulate API call
        return {
            'source': 'google_knowledge',
            'verified': True,
            'confidence': 0.7,
            'data': {'entities': ['Sample entity']}
        }

    async def _verify_fact_check(self, request: VerificationRequest):
        """Verify using Fact Check API"""
        await asyncio.sleep(1.2)  # Simulate API call
        return {
            'source': 'fact_check_api',
            'verified': False,
            'confidence': 0.6,
            'data': {'claims': []}
        }

    async def _verify_ecommerce(self, request: VerificationRequest):
        """Verify ecommerce-specific information"""
        await asyncio.sleep(0.3)  # Simulate API call
        return {
            'source': 'ecommerce_verifier',
            'verified': True,
            'confidence': 0.9,
            'data': {'product_available': True}
        }

    async def _verify_banking(self, request: VerificationRequest):
        """Verify banking-specific information"""
        await asyncio.sleep(0.6)  # Simulate API call
        return {
            'source': 'banking_verifier',
            'verified': True,
            'confidence': 0.85,
            'data': {'account_status': 'active'}
        }

    async def _verify_insurance(self, request: VerificationRequest):
        """Verify insurance-specific information"""
        await asyncio.sleep(0.7)  # Simulate API call
        return {
            'source': 'insurance_verifier',
            'verified': True,
            'confidence': 0.8,
            'data': {'policy_status': 'active'}
        }

    async def _verify_generic(self, route: VerificationRoute, request: VerificationRequest):
        """Generic verification method"""
        await asyncio.sleep(route.last_response_time)  # Simulate API call
        return {
            'source': route.source_name,
            'verified': True,
            'confidence': route.performance_score,
            'data': {'generic_verification': True}
        }

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        return {
            'metrics': self.metrics,
            'source_health': dict(self.source_health),
            'circuit_breaker_status': {
                source: circuit['state'] 
                for source, circuit in self.circuit_breaker.items()
            },
            'current_load': dict(self.current_load),
            'route_performance': {
                name: {
                    'performance_score': route.performance_score,
                    'success_rate': route.success_rate,
                    'last_response_time': route.last_response_time
                }
                for name, route in self.routes.items()
            }
        }

    def update_route_configuration(self, source_name: str, **kwargs):
        """Update route configuration"""
        if source_name in self.routes:
            route = self.routes[source_name]
            for key, value in kwargs.items():
                if hasattr(route, key):
                    setattr(route, key, value)

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all verification sources"""
        health_status = {}
        
        for source_name, route in self.routes.items():
            if not route.enabled:
                health_status[source_name] = {'status': 'disabled'}
                continue
            
            try:
                # Quick health check
                test_request = VerificationRequest(
                    query="health check",
                    response="health check response",
                    domain="general",
                    priority=VerificationPriority.LOW,
                    verification_type=VerificationType.REAL_TIME,
                    context={},
                    timeout=2.0
                )
                
                result = await self._execute_verification(route, test_request)
                health_status[source_name] = {
                    'status': 'healthy',
                    'response_time': route.last_response_time,
                    'performance_score': route.performance_score
                }
            except Exception as e:
                health_status[source_name] = {
                    'status': 'unhealthy',
                    'error': str(e)
                }
        
        return health_status 