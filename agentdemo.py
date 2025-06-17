#!/usr/bin/env python3
"""
Educational Insurance Case Evaluation with Anthropic MCP Protocol
================================================================

An interactive Streamlit application demonstrating different levels of AI agents
for insurance case evaluation using the MCP (Model Context Protocol).

Requirements:
    pip install streamlit anthropic mcp typing-extensions pydantic plotly

To run:
    streamlit run insurance_mcp_education.py

Environment:
    export ANTHROPIC_API_KEY="your-api-key"
"""

import os
import json
import asyncio
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import anthropic
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import Tool, TextContent, ImageContent, EmbeddedResource
import time
import numpy as np


# ============================================================================
# Configuration
# ============================================================================

# Page config
st.set_page_config(
    page_title="AI Agent Levels: Insurance MCP Demo",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if 'evaluation_history' not in st.session_state:
    st.session_state.evaluation_history = []
if 'agent_conversations' not in st.session_state:
    st.session_state.agent_conversations = {}
if 'tool_calls' not in st.session_state:
    st.session_state.tool_calls = []

# Try to get API key from environment first, then from Streamlit secrets
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# If not in environment, try Streamlit secrets (but don't error if secrets file doesn't exist)
if not ANTHROPIC_API_KEY:
    try:
        ANTHROPIC_API_KEY = st.secrets.get("ANTHROPIC_API_KEY")
    except FileNotFoundError:
        pass

if not ANTHROPIC_API_KEY:
    st.error("⚠️ Please set ANTHROPIC_API_KEY environment variable or add it to Streamlit secrets")
    st.info("""
    **Option 1: Environment Variable**
    ```bash
    export ANTHROPIC_API_KEY="your-api-key"
    streamlit run insurance_mcp_education.py
    ```
    
    **Option 2: Streamlit Secrets**
    Create `.streamlit/secrets.toml` in your project directory:
    ```toml
    ANTHROPIC_API_KEY = "your-api-key"
    ```
    """)
    st.stop()

# Initialize Anthropic client
client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


# ============================================================================
# Insurance Domain Models (Same as before)
# ============================================================================

class ClaimType(Enum):
    AUTO = "auto"
    PROPERTY = "property"
    HEALTH = "health"
    LIABILITY = "liability"


class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class InsuranceClaim:
    """Insurance claim data"""
    claim_id: str
    claim_type: ClaimType
    amount: float
    date_filed: datetime
    description: str
    policy_number: str
    claimant_history: List[Dict[str, Any]] = field(default_factory=list)
    documents: List[str] = field(default_factory=list)
    risk_indicators: List[str] = field(default_factory=list)
    status: str = "pending"


@dataclass
class EvaluationResult:
    """Evaluation result from an agent"""
    claim_id: str
    recommendation: str  # "approve", "deny", "investigate"
    confidence: float
    risk_level: RiskLevel
    reasons: List[str]
    suggested_payout: Optional[float] = None
    red_flags: List[str] = field(default_factory=list)
    evaluator: str = "unknown"
    processing_time: float = 0.0
    tool_usage: List[Dict[str, Any]] = field(default_factory=list)


# ============================================================================
# Enhanced MCP Tools with Logging
# ============================================================================

class InsuranceMCPTools:
    """MCP-compatible tools for insurance evaluation with educational logging"""
    
    @staticmethod
    def get_tools() -> List[Tool]:
        """Return MCP tool definitions"""
        return [
            Tool(
                name="check_policy_coverage",
                description="Check if a claim type is covered by the policy",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "policy_number": {"type": "string"},
                        "claim_type": {"type": "string"}
                    },
                    "required": ["policy_number", "claim_type"]
                }
            ),
            Tool(
                name="calculate_risk_score",
                description="Calculate risk score for a claim based on multiple factors",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "claim_amount": {"type": "number"},
                        "claim_history_count": {"type": "integer"},
                        "days_since_incident": {"type": "integer"},
                        "risk_indicators": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    },
                    "required": ["claim_amount"]
                }
            ),
            Tool(
                name="verify_documents",
                description="Verify if all required documents are present",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "claim_type": {"type": "string"},
                        "provided_documents": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    },
                    "required": ["claim_type", "provided_documents"]
                }
            ),
            Tool(
                name="check_fraud_patterns",
                description="Check for common fraud patterns in the claim",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "claim_description": {"type": "string"},
                        "claim_amount": {"type": "number"},
                        "claimant_history": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "date": {"type": "string"},
                                    "amount": {"type": "number"}
                                }
                            }
                        }
                    },
                    "required": ["claim_description", "claim_amount"]
                }
            ),
            Tool(
                name="investigate_claim",
                description="Perform detailed investigation of a claim",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "claim_id": {"type": "string"},
                        "investigation_type": {
                            "type": "string",
                            "enum": ["social_media", "witnesses", "expert_assessment", "site_inspection"]
                        }
                    },
                    "required": ["claim_id", "investigation_type"]
                }
            )
        ]
    
    @staticmethod
    def get_anthropic_tools() -> List[Dict[str, Any]]:
        """Convert MCP tools to Anthropic format"""
        mcp_tools = InsuranceMCPTools.get_tools()
        anthropic_tools = []
        
        for tool in mcp_tools:
            anthropic_tools.append({
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema
            })
        
        return anthropic_tools
    
    @staticmethod
    async def execute_tool(name: str, arguments: Dict[str, Any], log_callback=None) -> Dict[str, Any]:
        """Execute a tool and return results with logging"""
        
        # Log tool execution
        if log_callback:
            log_callback({
                "timestamp": datetime.now().isoformat(),
                "tool": name,
                "input": arguments,
                "status": "executing"
            })
        
        result = {}
        
        if name == "check_policy_coverage":
            # Simulate policy coverage check
            result = {
                "covered": True,
                "coverage_limit": 100000,
                "deductible": 1000,
                "exclusions": []
            }
        
        elif name == "calculate_risk_score":
            # Calculate risk based on inputs
            score = 0.0
            factors = []
            
            if arguments.get("claim_amount", 0) > 20000:
                score += 0.3
                factors.append("High claim amount")
            if arguments.get("claim_history_count", 0) > 2:
                score += 0.2
                factors.append("Multiple previous claims")
            if arguments.get("days_since_incident", 30) < 7:
                score += 0.1
                factors.append("Recent incident")
            
            risk_indicators = arguments.get("risk_indicators", [])
            score += len(risk_indicators) * 0.1
            if risk_indicators:
                factors.append(f"{len(risk_indicators)} risk indicators present")
            
            result = {
                "risk_score": min(score, 1.0),
                "risk_factors": factors,
                "calculation_details": {
                    "claim_amount_factor": 0.3 if arguments.get("claim_amount", 0) > 20000 else 0,
                    "history_factor": 0.2 if arguments.get("claim_history_count", 0) > 2 else 0,
                    "recency_factor": 0.1 if arguments.get("days_since_incident", 30) < 7 else 0,
                    "indicators_factor": len(risk_indicators) * 0.1
                }
            }
        
        elif name == "verify_documents":
            # Check document completeness
            required = {
                "auto": ["police_report", "photos", "repair_estimate"],
                "property": ["photos", "repair_estimate", "proof_of_ownership"],
                "health": ["medical_records", "bills", "prescription"],
                "liability": ["incident_report", "witness_statements"]
            }
            
            claim_type = arguments.get("claim_type", "auto")
            provided = arguments.get("provided_documents", [])
            required_docs = required.get(claim_type, [])
            missing = [doc for doc in required_docs if doc not in provided]
            
            result = {
                "complete": len(missing) == 0,
                "missing_documents": missing,
                "verification_status": "verified" if len(missing) == 0 else "incomplete",
                "required_documents": required_docs,
                "provided_documents": provided
            }
        
        elif name == "check_fraud_patterns":
            # Analyze for fraud patterns
            description = arguments.get("claim_description", "").lower()
            suspicious_terms = ["suddenly", "immediately", "total loss", "stolen", "disappeared"]
            found_terms = [term for term in suspicious_terms if term in description]
            
            history = arguments.get("claimant_history", [])
            frequent_claims = len(history) > 3
            
            fraud_score = len(found_terms) * 0.2 + (0.3 if frequent_claims else 0)
            
            result = {
                "fraud_score": min(fraud_score, 1.0),
                "suspicious_patterns": found_terms,
                "frequent_claimant": frequent_claims,
                "recommendation": "investigate" if fraud_score > 0.5 else "proceed",
                "analysis_details": {
                    "suspicious_terms_found": len(found_terms),
                    "total_previous_claims": len(history),
                    "fraud_score_calculation": f"{len(found_terms)} * 0.2 + {'0.3' if frequent_claims else '0'} = {fraud_score}"
                }
            }
        
        elif name == "investigate_claim":
            # Simulate investigation
            investigation_results = {
                "social_media": {
                    "contradictions_found": False,
                    "relevant_posts": 0,
                    "summary": "No contradictory social media activity found"
                },
                "witnesses": {
                    "witnesses_found": 2,
                    "statements_consistent": True,
                    "summary": "2 witnesses corroborate the claim"
                },
                "expert_assessment": {
                    "assessment_complete": True,
                    "findings": "Damage consistent with claimed incident",
                    "expert_confidence": 0.85
                },
                "site_inspection": {
                    "inspection_complete": True,
                    "evidence_found": True,
                    "physical_evidence": ["Skid marks", "Vehicle damage patterns"]
                }
            }
            
            result = investigation_results.get(
                arguments.get("investigation_type", "witnesses"),
                {"error": "Unknown investigation type"}
            )
        
        # Log tool result
        if log_callback:
            log_callback({
                "timestamp": datetime.now().isoformat(),
                "tool": name,
                "output": result,
                "status": "completed"
            })
        
        return result


# ============================================================================
# Enhanced Agent Classes with Educational Features
# ============================================================================

class ReactiveInsuranceAgent:
    """
    Level 1: Reactive Agent
    - No LLM usage
    - Simple rule-based responses
    - No memory or learning
    """
    
    def __init__(self):
        self.name = "Reactive Agent"
        self.description = "Simple rule-based agent with no AI capabilities"
        self.capabilities = [
            "❌ No LLM usage",
            "❌ No tool usage",
            "✅ Fast response time",
            "✅ Deterministic outcomes",
            "❌ No contextual understanding"
        ]
    
    def evaluate_claim(self, claim: InsuranceClaim) -> EvaluationResult:
        """Simple rule-based evaluation"""
        start_time = time.time()
        
        # Simple rules
        rules_applied = []
        
        if claim.amount > 10000:
            recommendation = "investigate"
            risk_level = RiskLevel.HIGH
            rules_applied.append(f"Rule: Amount > $10,000 → Investigate")
            reasons = [f"High claim amount: ${claim.amount:,.2f}"]
            confidence = 0.3
        else:
            recommendation = "approve"
            risk_level = RiskLevel.LOW
            rules_applied.append(f"Rule: Amount ≤ $10,000 → Approve")
            reasons = ["Low claim amount"]
            confidence = 0.3
            
        # Log decision process
        st.session_state.agent_conversations[self.name] = {
            "thought_process": rules_applied,
            "decision_tree": {
                "claim_amount": claim.amount,
                "threshold": 10000,
                "decision": recommendation
            }
        }
        
        return EvaluationResult(
            claim_id=claim.claim_id,
            recommendation=recommendation,
            confidence=confidence,
            risk_level=risk_level,
            reasons=reasons,
            suggested_payout=claim.amount * 0.8 if recommendation == "approve" else None,
            evaluator=self.name,
            processing_time=time.time() - start_time
        )


class AssistantInsuranceAgent:
    """
    Level 2: Assistant Agent
    - Uses LLM with MCP tools
    - Responds to requests
    - No autonomous behavior
    """
    
    def __init__(self):
        self.name = "Assistant Agent (Claude with MCP)"
        self.description = "LLM-powered agent that uses tools when asked"
        self.capabilities = [
            "✅ LLM reasoning",
            "✅ MCP tool usage",
            "✅ Structured responses",
            "❌ No proactive investigation",
            "❌ No memory"
        ]
        self.tools = InsuranceMCPTools()
    
    async def evaluate_claim(self, claim: InsuranceClaim) -> EvaluationResult:
        """Evaluate using Claude with MCP tools"""
        start_time = time.time()
        tool_usage_log = []
        
        # Prepare claim summary
        claim_summary = f"""
        Claim ID: {claim.claim_id}
        Type: {claim.claim_type.value}
        Amount: ${claim.amount:,.2f}
        Description: {claim.description}
        Filed: {claim.date_filed.strftime('%Y-%m-%d')}
        Previous claims: {len(claim.claimant_history)}
        Documents provided: {', '.join(claim.documents)}
        Risk indicators: {', '.join(claim.risk_indicators)}
        """
        
        # Track conversation
        conversation_log = [{
            "role": "system",
            "content": "Insurance claim evaluator using MCP tools"
        }, {
            "role": "user",
            "content": claim_summary
        }]
        
        # Create message with tools
        with st.spinner(f"🤖 {self.name} is analyzing..."):
            message = client.messages.create(
                model="claude-3-7-sonnet-20250219",
                max_tokens=1000,
                temperature=0.3,
                tools=self.tools.get_anthropic_tools(),
                messages=[
                    {
                        "role": "user",
                        "content": f"""You are an insurance claim evaluator. Evaluate this claim and provide a recommendation.
                        
{claim_summary}

Use the available tools to:
1. Check if the claim is covered by the policy
2. Calculate the risk score
3. Verify documents are complete
4. Check for fraud patterns

Based on your analysis, provide:
- Recommendation: approve, deny, or investigate
- Confidence level (0-1)
- Risk level: low, medium, high, or critical
- Reasons for your decision
- Red flags if any

Think step by step and explain your reasoning."""
                    }
                ]
            )
        
        # Process tool calls
        risk_score = 0.5
        red_flags = []
        reasons = []
        
        # Log Claude's response
        conversation_log.append({
            "role": "assistant",
            "content": "Processing claim with MCP tools..."
        })
        
        if message.content:
            for content in message.content:
                if hasattr(content, 'type') and content.type == 'tool_use':
                    # Execute tool
                    def log_tool_call(log_entry):
                        tool_usage_log.append(log_entry)
                    
                    result = await self.tools.execute_tool(
                        content.name,
                        content.input,
                        log_callback=log_tool_call
                    )
                    
                    # Process results
                    if content.name == "calculate_risk_score":
                        risk_score = result.get("risk_score", 0.5)
                        reasons.extend(result.get("risk_factors", []))
                    
                    elif content.name == "check_fraud_patterns":
                        if result.get("fraud_score", 0) > 0.5:
                            red_flags.append("High fraud score")
                            red_flags.extend(result.get("suspicious_patterns", []))
                    
                    elif content.name == "verify_documents":
                        if not result.get("complete"):
                            reasons.append(f"Missing documents: {', '.join(result.get('missing_documents', []))}")
        
        # Extract final recommendation from Claude's response
        response_text = message.content[0].text if message.content else ""
        conversation_log.append({
            "role": "assistant",
            "content": response_text
        })
        
        # Parse recommendation
        recommendation = "investigate"
        if "approve" in response_text.lower():
            recommendation = "approve"
        elif "deny" in response_text.lower():
            recommendation = "deny"
        
        # Determine risk level
        if risk_score > 0.7:
            risk_level = RiskLevel.CRITICAL
        elif risk_score > 0.5:
            risk_level = RiskLevel.HIGH
        elif risk_score > 0.3:
            risk_level = RiskLevel.MEDIUM
        else:
            risk_level = RiskLevel.LOW
        
        # Store conversation for display
        st.session_state.agent_conversations[self.name] = {
            "conversation": conversation_log,
            "tool_usage": tool_usage_log,
            "reasoning": response_text
        }
        
        return EvaluationResult(
            claim_id=claim.claim_id,
            recommendation=recommendation,
            confidence=0.7,
            risk_level=risk_level,
            reasons=reasons[:3] if reasons else ["Evaluated using MCP tools"],
            red_flags=red_flags,
            evaluator=self.name,
            processing_time=time.time() - start_time,
            tool_usage=tool_usage_log
        )


class AutonomousInsuranceAgent:
    """
    Level 3: Autonomous Agent
    - Proactive investigation
    - Maintains conversation memory
    - Learns from patterns
    """
    
    def __init__(self):
        self.name = "Autonomous Agent (Claude with Memory)"
        self.description = "Proactive agent with memory and pattern recognition"
        self.capabilities = [
            "✅ LLM reasoning",
            "✅ MCP tool usage",
            "✅ Proactive investigation",
            "✅ Pattern learning",
            "✅ Memory across claims"
        ]
        self.tools = InsuranceMCPTools()
        self.memory = []
        self.pattern_insights = {}
    
    async def evaluate_claim(self, claim: InsuranceClaim) -> Tuple[EvaluationResult, List[str]]:
        """Autonomously evaluate with memory and learning"""
        start_time = time.time()
        actions_taken = []
        tool_usage_log = []
        
        # Add to memory
        self.memory.append({
            "timestamp": datetime.now(),
            "claim_id": claim.claim_id,
            "type": claim.claim_type.value,
            "amount": claim.amount
        })
        
        # Prepare context with memory
        memory_context = ""
        if len(self.memory) > 1:
            recent_claims = self.memory[-5:]
            memory_context = f"""
Previous evaluations:
{chr(10).join([f"- {m['type']} claim for ${m['amount']:,.2f}" for m in recent_claims[:-1]])}

Patterns observed:
- Average claim amount: ${sum(m['amount'] for m in recent_claims) / len(recent_claims):,.2f}
- Claim frequency: {len(recent_claims)} in recent history
- Pattern insights: {json.dumps(self.pattern_insights, indent=2)}
"""
        
        claim_summary = f"""
        Claim ID: {claim.claim_id}
        Type: {claim.claim_type.value}
        Amount: ${claim.amount:,.2f}
        Description: {claim.description}
        Filed: {claim.date_filed.strftime('%Y-%m-%d')}
        Previous claims: {len(claim.claimant_history)}
        Documents: {', '.join(claim.documents)}
        Risk indicators: {', '.join(claim.risk_indicators)}
        """
        
        # Create autonomous evaluation prompt
        with st.spinner(f"🤖 {self.name} is proactively investigating..."):
            message = client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=2000,
                temperature=0.4,
                tools=self.tools.get_anthropic_tools(),
                system="""You are an autonomous insurance claim evaluator with the ability to:
1. Proactively investigate claims
2. Identify patterns across multiple claims
3. Make decisions based on accumulated knowledge
4. Escalate complex cases when confidence is low

You should think step-by-step and use tools proactively to gather all necessary information.
Explain your reasoning and what patterns you notice.""",
                messages=[
                    {
                        "role": "user",
                        "content": f"""Evaluate this insurance claim autonomously. 

{memory_context}

Current claim:
{claim_summary}

Proactively investigate using available tools. Consider:
1. Is this claim consistent with patterns you've seen?
2. What additional investigation might reveal important information?
3. Are there any red flags that warrant deeper investigation?
4. Should this be escalated to a senior evaluator?

Think through your evaluation process step by step, using tools as needed.
Explain what you're doing and why."""
                    }
                ]
            )
        
        # Track actions and extract insights
        confidence = 0.8
        reasons = []
        red_flags = []
        risk_level = RiskLevel.MEDIUM
        recommendation = "investigate"
        reasoning_log = []
        
        # Process response and tool usage
        for content in message.content:
            if hasattr(content, 'type') and content.type == 'tool_use':
                actions_taken.append(f"Used tool: {content.name}")
                
                # Execute tool
                def log_tool_call(log_entry):
                    tool_usage_log.append(log_entry)
                
                result = await self.tools.execute_tool(
                    content.name, 
                    content.input,
                    log_callback=log_tool_call
                )
                
                # Process based on tool type
                if content.name == "investigate_claim":
                    actions_taken.append("Proactive investigation initiated")
                    if any("contradict" in str(v).lower() for v in result.values()):
                        red_flags.append("Investigation found contradictions")
                        confidence -= 0.2
                
                elif content.name == "check_fraud_patterns":
                    fraud_score = result.get("fraud_score", 0)
                    if fraud_score > 0.5:
                        red_flags.extend(result.get("suspicious_patterns", []))
                        confidence -= 0.1
                        risk_level = RiskLevel.HIGH
            
            elif hasattr(content, 'text'):
                reasoning_log.append(content.text)
                
                # Extract insights from Claude's reasoning
                text = content.text.lower()
                if "high risk" in text:
                    risk_level = RiskLevel.HIGH
                elif "low risk" in text:
                    risk_level = RiskLevel.LOW
                
                if "approve" in text and "recommend" in text:
                    recommendation = "approve"
                elif "deny" in text and "recommend" in text:
                    recommendation = "deny"
                
                # Extract reasons
                if "because" in text:
                    reason_start = text.find("because")
                    reason_text = text[reason_start:reason_start+100].split('.')[0]
                    reasons.append(reason_text)
        
        # Update pattern insights
        pattern_key = f"{claim.claim_type.value}_{risk_level.value}"
        if pattern_key not in self.pattern_insights:
            self.pattern_insights[pattern_key] = {"count": 0, "outcomes": {}}
        
        self.pattern_insights[pattern_key]["count"] += 1
        self.pattern_insights[pattern_key]["outcomes"][recommendation] = \
            self.pattern_insights[pattern_key]["outcomes"].get(recommendation, 0) + 1
        
        actions_taken.append(f"Updated pattern insights for {pattern_key}")
        
        # Low confidence triggers escalation
        if confidence < 0.6:
            actions_taken.append("Escalating to senior evaluator due to low confidence")
            reasons.append("Requires senior review")
        
        # Store conversation for display
        st.session_state.agent_conversations[self.name] = {
            "memory_context": memory_context,
            "reasoning": reasoning_log,
            "tool_usage": tool_usage_log,
            "patterns": self.pattern_insights,
            "actions": actions_taken
        }
        
        return EvaluationResult(
            claim_id=claim.claim_id,
            recommendation=recommendation,
            confidence=confidence,
            risk_level=risk_level,
            reasons=reasons[:3] if reasons else ["Autonomous evaluation completed"],
            red_flags=red_flags,
            evaluator=self.name,
            processing_time=time.time() - start_time,
            tool_usage=tool_usage_log
        ), actions_taken


class MultiAgentInsuranceSystem:
    """
    Level 4: Multi-Agent System
    - Multiple specialized Claude instances
    - Collaborative decision making
    - Emergent consensus
    """
    
    def __init__(self):
        self.name = "Multi-Agent System (Claude Specialists)"
        self.description = "Multiple specialized agents working together"
        self.capabilities = [
            "✅ Multiple perspectives",
            "✅ Specialist knowledge",
            "✅ Consensus building",
            "✅ Senior arbitration",
            "✅ Comprehensive analysis"
        ]
        self.tools = InsuranceMCPTools()
    
    async def evaluate_claim(self, claim: InsuranceClaim) -> Tuple[EvaluationResult, Dict[str, Any]]:
        """Multi-agent collaborative evaluation"""
        start_time = time.time()
        evaluation_log = {
            "agents": ["Fraud Specialist", "Risk Analyst", "Customer Advocate"],
            "consensus": False,
            "deliberation_rounds": 0,
            "agent_conversations": {}
        }
        
        claim_summary = f"""
        Claim ID: {claim.claim_id}
        Type: {claim.claim_type.value}
        Amount: ${claim.amount:,.2f}
        Description: {claim.description}
        Documents: {', '.join(claim.documents)}
        Risk indicators: {', '.join(claim.risk_indicators)}
        """
        
        # Get evaluations from different specialist perspectives
        evaluations = {}
        
        with st.spinner("🤖 Specialists are evaluating..."):
            # 1. Fraud Specialist
            fraud_eval, fraud_conv = await self._get_specialist_evaluation(
                claim_summary,
                "fraud detection specialist",
                "Focus on identifying potential fraud patterns, suspicious timing, and inconsistencies."
            )
            evaluations["Fraud Specialist"] = fraud_eval
            evaluation_log["agent_conversations"]["Fraud Specialist"] = fraud_conv
            
            # 2. Risk Analyst
            risk_eval, risk_conv = await self._get_specialist_evaluation(
                claim_summary,
                "risk analysis expert",
                "Focus on calculating overall risk, considering claim amount, history, and documentation."
            )
            evaluations["Risk Analyst"] = risk_eval
            evaluation_log["agent_conversations"]["Risk Analyst"] = risk_conv
            
            # 3. Customer Advocate
            customer_eval, customer_conv = await self._get_specialist_evaluation(
                claim_summary,
                "customer advocate",
                "Focus on fair treatment, customer history, and giving benefit of the doubt where reasonable."
            )
            evaluations["Customer Advocate"] = customer_eval
            evaluation_log["agent_conversations"]["Customer Advocate"] = customer_conv
        
        # Check for consensus
        recommendations = [e["recommendation"] for e in evaluations.values()]
        consensus_recommendation = max(set(recommendations), key=recommendations.count)
        
        if recommendations.count(consensus_recommendation) >= 2:
            evaluation_log["consensus"] = True
            
            # Average confidence
            avg_confidence = sum(e["confidence"] for e in evaluations.values()) / 3
            
            # Combine insights
            all_reasons = []
            all_red_flags = []
            for agent, eval_data in evaluations.items():
                all_reasons.extend([f"[{agent}] {r}" for r in eval_data.get("reasons", [])])
                all_red_flags.extend(eval_data.get("red_flags", []))
            
            # Determine risk level (take highest)
            risk_levels = [e["risk_level"] for e in evaluations.values()]
            risk_order = [RiskLevel.CRITICAL, RiskLevel.HIGH, RiskLevel.MEDIUM, RiskLevel.LOW]
            final_risk = RiskLevel.MEDIUM
            for risk in risk_order:
                if risk.value in risk_levels:
                    final_risk = risk
                    break
            
            result = EvaluationResult(
                claim_id=claim.claim_id,
                recommendation=consensus_recommendation,
                confidence=avg_confidence,
                risk_level=final_risk,
                reasons=all_reasons[:5],
                red_flags=list(set(all_red_flags)),
                evaluator=self.name,
                processing_time=time.time() - start_time
            )
        else:
            # No consensus - senior arbitration needed
            evaluation_log["consensus"] = False
            with st.spinner("🤖 Senior evaluator making final decision..."):
                result, arbitration_conv = await self._senior_arbitration(claim_summary, evaluations)
                evaluation_log["agent_conversations"]["Senior Arbitrator"] = arbitration_conv
        
        evaluation_log["evaluations"] = evaluations
        
        # Store for display
        st.session_state.agent_conversations[self.name] = evaluation_log
        
        return result, evaluation_log
    
    async def _get_specialist_evaluation(self, claim_summary: str, role: str, focus: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Get evaluation from a specialist perspective"""
        tool_usage_log = []
        
        def log_tool_call(log_entry):
            tool_usage_log.append(log_entry)
        
        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            temperature=0.3,
            tools=self.tools.get_anthropic_tools(),
            system=f"You are a {role} evaluating insurance claims. {focus}",
            messages=[
                {
                    "role": "user",
                    "content": f"""Evaluate this claim from your specialist perspective:

{claim_summary}

Provide:
1. Recommendation (approve/deny/investigate)
2. Confidence level (0-1)
3. Risk assessment (low/medium/high/critical)
4. Top 3 reasons for your decision
5. Any red flags you identify

Use tools as needed for your analysis. Explain your specialist perspective."""
                }
            ]
        )
        
        # Parse response
        response_text = message.content[0].text if message.content else ""
        
        # Extract key information
        recommendation = "investigate"
        if "approve" in response_text.lower() and "recommend" in response_text.lower():
            recommendation = "approve"
        elif "deny" in response_text.lower() and "recommend" in response_text.lower():
            recommendation = "deny"
        
        # Extract confidence
        confidence = 0.7
        if "high confidence" in response_text.lower():
            confidence = 0.9
        elif "low confidence" in response_text.lower():
            confidence = 0.4
        
        # Extract risk level
        risk_level = "medium"
        if "critical" in response_text.lower() or "very high risk" in response_text.lower():
            risk_level = "critical"
        elif "high risk" in response_text.lower():
            risk_level = "high"
        elif "low risk" in response_text.lower():
            risk_level = "low"
        
        conversation = {
            "role": role,
            "focus": focus,
            "response": response_text,
            "tool_usage": tool_usage_log
        }
        
        return {
            "recommendation": recommendation,
            "confidence": confidence,
            "risk_level": risk_level,
            "reasons": [f"{role} assessment completed"],
            "red_flags": []
        }, conversation
    
    async def _senior_arbitration(self, claim_summary: str, evaluations: Dict[str, Dict]) -> Tuple[EvaluationResult, Dict[str, Any]]:
        """Senior evaluator makes final decision when no consensus"""
        
        # Prepare evaluation summary
        eval_summary = "\n".join([
            f"{agent}: {data['recommendation']} (confidence: {data['confidence']})"
            for agent, data in evaluations.items()
        ])
        
        message = client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=1000,
            temperature=0.2,
            system="You are a senior insurance evaluator making final decisions when specialist agents disagree.",
            messages=[
                {
                    "role": "user",
                    "content": f"""The specialist agents could not reach consensus on this claim:

{claim_summary}

Agent evaluations:
{eval_summary}

As the senior evaluator, make the final decision. Consider all perspectives and provide:
1. Final recommendation
2. Confidence in your decision
3. Risk level assessment
4. Key reasons for your decision

Explain your arbitration process."""
                }
            ]
        )
        
        # Extract decision
        response_text = message.content[0].text if message.content else ""
        
        recommendation = "investigate"
        if "approve" in response_text.lower():
            recommendation = "approve"
        elif "deny" in response_text.lower():
            recommendation = "deny"
        
        conversation = {
            "role": "Senior Arbitrator",
            "evaluations_summary": eval_summary,
            "decision_process": response_text
        }
        
        return EvaluationResult(
            claim_id="CLM-2024-001",  # Fix hardcoded ID
            recommendation=recommendation,
            confidence=0.85,
            risk_level=RiskLevel.MEDIUM,
            reasons=["Senior arbitration: no consensus among specialists"],
            evaluator="Senior Evaluator"
        ), conversation


# ============================================================================
# Streamlit UI
# ============================================================================

def main():
    st.title("🤖 AI Agent Levels: Insurance Claims with MCP Protocol")
    st.markdown("""
    This educational demo shows how different levels of AI agents evaluate insurance claims using the Model Context Protocol (MCP).
    Watch how agents progress from simple rules to complex multi-agent systems!
    """)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("📋 Create Test Claim")
        
        claim_type = st.selectbox(
            "Claim Type",
            options=[ct.value for ct in ClaimType],
            format_func=lambda x: x.title()
        )
        
        claim_amount = st.number_input(
            "Claim Amount ($)",
            min_value=1000,
            max_value=100000,
            value=15000,
            step=1000
        )
        
        claim_description = st.text_area(
            "Claim Description",
            value="Vehicle collision at intersection. Sudden brake failure caused accident.",
            height=100
        )
        
        # Document selection
        st.subheader("📄 Documents Provided")
        doc_options = {
            "auto": ["police_report", "photos", "repair_estimate"],
            "property": ["photos", "repair_estimate", "proof_of_ownership"],
            "health": ["medical_records", "bills", "prescription"],
            "liability": ["incident_report", "witness_statements"]
        }
        
        selected_docs = st.multiselect(
            "Select documents",
            options=doc_options.get(claim_type, []),
            default=["police_report", "photos"] if claim_type == "auto" else []
        )
        
        # Risk indicators
        risk_indicators = st.multiselect(
            "Risk Indicators",
            options=["multiple_claims", "high_amount", "recent_incident", "suspicious_timing"],
            default=["multiple_claims", "high_amount"]
        )
        
        # Previous claims
        num_previous_claims = st.slider("Number of Previous Claims", 0, 5, 2)
        
        if st.button("🚀 Evaluate Claim", type="primary", use_container_width=True):
            # Create claim
            claim = InsuranceClaim(
                claim_id=f"CLM-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                claim_type=ClaimType(claim_type),
                amount=claim_amount,
                date_filed=datetime.now() - timedelta(days=3),
                description=claim_description,
                policy_number="POL-123456",
                claimant_history=[
                    {"date": f"2023-{i:02d}-01", "amount": 5000 + i * 1000}
                    for i in range(1, num_previous_claims + 1)
                ],
                documents=selected_docs,
                risk_indicators=risk_indicators
            )
            
            # Run evaluations
            asyncio.run(evaluate_all_agents(claim))
    
    # Display results
    if st.session_state.evaluation_history:
        display_results()


async def evaluate_all_agents(claim: InsuranceClaim):
    """Run all agent evaluations"""
    st.session_state.agent_conversations = {}
    results = []
    
    # Create progress placeholder
    progress_placeholder = st.empty()
    
    # Level 1: Reactive Agent
    with progress_placeholder.container():
        st.info("🔄 Running Level 1: Reactive Agent...")
    reactive = ReactiveInsuranceAgent()
    result1 = reactive.evaluate_claim(claim)
    results.append(result1)
    
    # Level 2: Assistant Agent
    with progress_placeholder.container():
        st.info("🔄 Running Level 2: Assistant Agent...")
    assistant = AssistantInsuranceAgent()
    result2 = await assistant.evaluate_claim(claim)
    results.append(result2)
    
    # Level 3: Autonomous Agent
    with progress_placeholder.container():
        st.info("🔄 Running Level 3: Autonomous Agent...")
    autonomous = AutonomousInsuranceAgent()
    result3, actions = await autonomous.evaluate_claim(claim)
    results.append(result3)
    
    # Level 4: Multi-Agent System
    with progress_placeholder.container():
        st.info("🔄 Running Level 4: Multi-Agent System...")
    multi_agent = MultiAgentInsuranceSystem()
    result4, log = await multi_agent.evaluate_claim(claim)
    results.append(result4)
    
    # Clear progress
    progress_placeholder.empty()
    
    # Store results
    st.session_state.evaluation_history = results
    st.success("✅ All evaluations complete!")


def display_results():
    """Display evaluation results with educational insights"""
    st.header("📊 Evaluation Results")
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Overview", 
        "🤖 Agent Details", 
        "🔧 Tool Usage", 
        "💭 Reasoning Process",
        "📚 Learning Points",
        "🏗️ Architecture"
    ])
    
    with tab1:
        display_overview()
    
    with tab2:
        display_agent_details()
    
    with tab3:
        display_tool_usage()
    
    with tab4:
        display_reasoning()
    
    with tab5:
        display_learning_points()
    
    with tab6:
        display_architecture()


def display_overview():
    """Display overview of all agent results"""
    results = st.session_state.evaluation_history
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Agents", len(results))
    
    with col2:
        recommendations = [r.recommendation for r in results]
        most_common = max(set(recommendations), key=recommendations.count)
        st.metric("Consensus", most_common.title())
    
    with col3:
        avg_confidence = sum(r.confidence for r in results) / len(results)
        st.metric("Avg Confidence", f"{avg_confidence:.2%}")
    
    with col4:
        avg_time = sum(r.processing_time for r in results) / len(results)
        st.metric("Avg Time", f"{avg_time:.2f}s")
    
    # Comparison chart
    df_results = pd.DataFrame([{
        'Agent': r.evaluator,
        'Recommendation': r.recommendation,
        'Confidence': r.confidence,
        'Risk Level': r.risk_level.value,
        'Processing Time': r.processing_time
    } for r in results])
    
    # Confidence comparison
    fig_confidence = px.bar(
        df_results, 
        x='Agent', 
        y='Confidence',
        title='Agent Confidence Levels',
        color='Recommendation',
        text='Confidence'
    )
    fig_confidence.update_traces(texttemplate='%{text:.2%}', textposition='outside')
    st.plotly_chart(fig_confidence, use_container_width=True)
    
    # Processing time comparison
    fig_time = px.bar(
        df_results,
        x='Agent',
        y='Processing Time',
        title='Processing Time by Agent Level',
        text='Processing Time'
    )
    fig_time.update_traces(texttemplate='%{text:.2f}s', textposition='outside')
    st.plotly_chart(fig_time, use_container_width=True)


def display_agent_details():
    """Display detailed information about each agent"""
    agents = [
        ReactiveInsuranceAgent(),
        AssistantInsuranceAgent(),
        AutonomousInsuranceAgent(),
        MultiAgentInsuranceSystem()
    ]
    
    for i, (agent, result) in enumerate(zip(agents, st.session_state.evaluation_history)):
        with st.expander(f"Level {i+1}: {agent.name}", expanded=i==0):
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown(f"**Description:** {agent.description}")
                st.markdown("**Capabilities:**")
                for cap in agent.capabilities:
                    st.write(cap)
            
            with col2:
                st.markdown("**Evaluation Result:**")
                st.write(f"✅ Recommendation: **{result.recommendation.upper()}**")
                st.write(f"📊 Confidence: **{result.confidence:.2%}**")
                st.write(f"⚠️ Risk Level: **{result.risk_level.value.upper()}**")
                
                if result.reasons:
                    st.markdown("**Reasons:**")
                    for reason in result.reasons:
                        st.write(f"- {reason}")
                
                if result.red_flags:
                    st.markdown("**🚩 Red Flags:**")
                    for flag in result.red_flags:
                        st.write(f"- {flag}")


def display_tool_usage():
    """Display MCP tool usage across agents"""
    st.subheader("🔧 MCP Tool Usage Analysis")
    
    # Collect all tool usage
    all_tools = []
    for result in st.session_state.evaluation_history:
        if hasattr(result, 'tool_usage') and result.tool_usage:
            for tool_call in result.tool_usage:
                all_tools.append({
                    'Agent': result.evaluator,
                    'Tool': tool_call.get('tool', 'Unknown'),
                    'Status': tool_call.get('status', 'Unknown')
                })
    
    if all_tools:
        df_tools = pd.DataFrame(all_tools)
        
        # Tool usage by agent
        fig_tools = px.histogram(
            df_tools,
            x='Agent',
            color='Tool',
            title='Tool Usage by Agent',
            barmode='group'
        )
        st.plotly_chart(fig_tools, use_container_width=True)
        
        # Detailed tool calls
        for agent in df_tools['Agent'].unique():
            agent_tools = df_tools[df_tools['Agent'] == agent]
            if not agent_tools.empty:
                st.markdown(f"**{agent} Tool Calls:**")
                for _, tool in agent_tools.iterrows():
                    st.write(f"- {tool['Tool']} ({tool['Status']})")
    else:
        st.info("No tool usage recorded (Level 1 agent doesn't use tools)")


def display_reasoning():
    """Display agent reasoning and conversations"""
    st.subheader("💭 Agent Reasoning Process")
    
    conversations = st.session_state.agent_conversations
    
    for agent_name, conv_data in conversations.items():
        with st.expander(f"{agent_name} Reasoning", expanded=False):
            if 'thought_process' in conv_data:
                # Level 1: Simple rules
                st.markdown("**Decision Process:**")
                for rule in conv_data['thought_process']:
                    st.write(f"- {rule}")
                
                if 'decision_tree' in conv_data:
                    st.json(conv_data['decision_tree'])
            
            elif 'reasoning' in conv_data:
                # Level 2 & 3: LLM reasoning
                st.markdown("**LLM Reasoning:**")
                if isinstance(conv_data['reasoning'], list):
                    for reasoning in conv_data['reasoning']:
                        st.text_area("Thought Process", reasoning, height=200)
                else:
                    st.text_area("Thought Process", conv_data['reasoning'], height=200)
                
                if 'memory_context' in conv_data:
                    st.markdown("**Memory Context:**")
                    st.text(conv_data['memory_context'])
                
                if 'patterns' in conv_data:
                    st.markdown("**Learned Patterns:**")
                    st.json(conv_data['patterns'])
            
            elif 'agent_conversations' in conv_data:
                # Level 4: Multi-agent
                st.markdown("**Multi-Agent Deliberation:**")
                
                for specialist, spec_conv in conv_data['agent_conversations'].items():
                    st.markdown(f"**{specialist}:**")
                    if 'response' in spec_conv:
                        st.text_area(f"{specialist} Analysis", spec_conv['response'], height=150)
                
                st.markdown(f"**Consensus Reached:** {'✅ Yes' if conv_data.get('consensus') else '❌ No'}")


def display_learning_points():
    """Display educational insights about agent levels"""
    st.subheader("📚 Key Learning Points")
    
    learning_points_dict = {
        "Level 1 - Reactive Agents": [
            "✅ **Pros:** Fast, deterministic, predictable",
            "❌ **Cons:** No context understanding, rigid rules, can't handle edge cases",
            "📝 **Use Case:** Simple, high-volume decisions with clear rules"
        ],
        "Level 2 - Assistant Agents": [
            "✅ **Pros:** LLM reasoning, tool usage, structured analysis",
            "❌ **Cons:** No memory, reactive only, no learning",
            "📝 **Use Case:** Complex analysis requiring reasoning but not continuity"
        ],
        "Level 3 - Autonomous Agents": [
            "✅ **Pros:** Proactive investigation, pattern learning, memory",
            "❌ **Cons:** More complex, requires state management",
            "📝 **Use Case:** Cases requiring historical context and pattern recognition"
        ],
        "Level 4 - Multi-Agent Systems": [
            "✅ **Pros:** Multiple perspectives, specialist knowledge, consensus building",
            "❌ **Cons:** Higher cost, longer processing time, coordination complexity",
            "📝 **Use Case:** High-stakes decisions requiring diverse expertise"
        ]
    }
    
    for agent_level_name, point_list in learning_points_dict.items():
        with st.expander(agent_level_name, expanded=True):
            for point in point_list:
                st.markdown(point)
    
    # MCP Protocol Benefits
    st.markdown("### 🔧 MCP Protocol Benefits")
    st.markdown("""
    - **Structured Tool Calling:** Type-safe, validated inputs and outputs
    - **Separation of Concerns:** Clear distinction between reasoning and tool execution
    - **Scalability:** Same protocol works from simple to complex agents
    - **Observability:** Easy to track and debug tool usage
    - **Extensibility:** Easy to add new tools without changing agent logic
    """)
    
    return  # Explicit return to end the function


def display_architecture():
    """Display detailed architecture information for each agent level"""
    st.subheader("🏗️ Agent Architecture Deep Dive")
    
    # Architecture selection
    selected_level = st.selectbox(
        "Select Agent Level to Explore",
        options=["Level 1: Reactive Agent", "Level 2: Assistant Agent", 
                 "Level 3: Autonomous Agent", "Level 4: Multi-Agent System"]
    )
    
    st.markdown("---")
    
    if "Level 1" in selected_level:
        display_level1_architecture()
    elif "Level 2" in selected_level:
        display_level2_architecture()
    elif "Level 3" in selected_level:
        display_level3_architecture()
    elif "Level 4" in selected_level:
        display_level4_architecture()
    
    # Common architectural patterns
    with st.expander("🔍 Common Architectural Patterns", expanded=False):
        st.markdown("""
        ### Design Patterns Used
        
        **1. Strategy Pattern**
        - Each agent implements the same evaluation interface
        - Allows swapping agents without changing the system
        
        **2. Observer Pattern**
        - Tool usage logging and callbacks
        - Monitoring agent decisions and actions
        
        **3. Chain of Responsibility**
        - Multi-agent system with escalation
        - Senior arbitration when needed
        
        **4. Memento Pattern**
        - Level 3 agents store state/memory
        - Pattern recognition across evaluations
        
        **5. Facade Pattern**
        - MCP tools provide simple interface to complex operations
        - Hides implementation details from agents
        """)
    
    # MCP Protocol Deep Dive
    with st.expander("🔧 MCP Protocol Deep Dive", expanded=False):
        st.markdown("""
        ### Model Context Protocol (MCP) Architecture
        
        **Protocol Layers:**
        
        1. **Tool Definition Layer**
           - JSON Schema for input validation
           - Strongly typed parameters
           - Clear tool descriptions
        
        2. **Registration Layer**
           - Tools registered with LLM at conversation start
           - LLM understands available capabilities
           - Dynamic tool discovery possible
        
        3. **Execution Layer**
           - LLM generates tool calls
           - Framework validates inputs
           - Tools executed in isolation
           - Results returned to LLM
        
        4. **Integration Layer**
           - LLM combines tool results
           - Reasoning over multiple tool outputs
           - Final decision synthesis
        
        **Benefits over Function Calling:**
        - Better type safety
        - Clearer separation of concerns
        - Easier testing and debugging
        - Protocol standardization
        """)
    
    # Performance Characteristics
    with st.expander("⚡ Performance Characteristics", expanded=False):
        st.markdown("""
        ### Performance Analysis by Level
        
        | Agent Level | Latency | Token Usage | Cost | Accuracy |
        |------------|---------|-------------|------|-----------|
        | Level 1 | ~10ms | 0 | $0 | Low |
        | Level 2 | ~2-3s | ~1.5k | ~$0.01 | Medium |
        | Level 3 | ~4-5s | ~3k | ~$0.03 | High |
        | Level 4 | ~8-10s | ~6k | ~$0.08 | Very High |
        
        **Scaling Considerations:**
        - Level 1: Can handle millions of requests
        - Level 2: Good for thousands per minute
        - Level 3: Stateful, requires memory management
        - Level 4: Best for critical decisions only
        
        **Optimization Strategies:**
        - Cache common tool results
        - Batch similar evaluations
        - Use Level 1 for initial filtering
        - Escalate only when necessary
        """)



def create_animated_architecture(level):
    """Create animated architecture visualizations"""
    
    if level == 1:
        # Animated flow chart
        fig = go.Figure()
        
        # Create animated bar chart showing decision speed
        categories = ['Input Processing', 'Rule Check', 'Decision', 'Output']
        times = [0.001, 0.002, 0.001, 0.001]  # in seconds
        colors = ['#64b5f6', '#fff59d', '#ffcc80', '#81c784']
        
        fig.add_trace(go.Bar(
            x=categories,
            y=times,
            marker_color=colors,
            text=[f'{t*1000:.1f}ms' for t in times],
            textposition='outside',
            name='Processing Time'
        ))
        
        fig.add_trace(go.Scatter(
            x=categories,
            y=[0.01, 0.01, 0.01, 0.01],
            mode='lines+markers',
            name='LLM Baseline',
            line=dict(color='red', dash='dash'),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title='Level 1: Reactive Agent - Lightning Fast Processing',
            yaxis_title='Time (seconds)',
            showlegend=True,
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0.1)',
            yaxis=dict(type='log', gridcolor='rgba(255,255,255,0.1)'),
            xaxis=dict(gridcolor='rgba(255,255,255,0.1)')
        )
        
        return fig
    
    elif level == 2:
        # Radar chart showing capabilities
        categories = ['Speed', 'Accuracy', 'Context', 'Learning', 'Autonomy', 'Cost']
        
        fig = go.Figure()
        
        # Level 1 comparison
        fig.add_trace(go.Scatterpolar(
            r=[10, 3, 1, 1, 1, 10],
            theta=categories,
            fill='toself',
            name='Level 1: Reactive',
            line_color='#ff6b6b',
            opacity=0.4
        ))
        
        # Level 2
        fig.add_trace(go.Scatterpolar(
            r=[6, 7, 8, 2, 3, 6],
            theta=categories,
            fill='toself',
            name='Level 2: Assistant',
            line_color='#4ecdc4'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 10],
                    gridcolor='rgba(255,255,255,0.2)'
                ),
                bgcolor='rgba(0,0,0,0)',
                angularaxis=dict(gridcolor='rgba(255,255,255,0.2)')
            ),
            showlegend=True,
            title="Level 2: Assistant Agent - Capability Profile",
            height=500,
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig
    
    elif level == 3:
        # Time series showing learning over time
        # Simulate confidence improvement over evaluations
        n_evaluations = 50
        x = list(range(n_evaluations))
        
        # Base confidence starts low and improves
        base_confidence = 0.6
        learning_curve = [base_confidence + (1 - base_confidence) * (1 - np.exp(-i/10)) + np.random.normal(0, 0.05) for i in x]
        
        # Pattern detections
        patterns_found = [0]
        for i in range(1, n_evaluations):
            if i % 7 == 0:
                patterns_found.append(patterns_found[-1] + 1)
            else:
                patterns_found.append(patterns_found[-1])
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Confidence Over Time', 'Patterns Discovered'),
            row_heights=[0.6, 0.4],
            vertical_spacing=0.1
        )
        
        # Confidence trace
        fig.add_trace(
            go.Scatter(x=x, y=learning_curve, mode='lines+markers',
                      name='Confidence', line=dict(color='#66bb6a', width=3),
                      marker=dict(size=6)),
            row=1, col=1
        )
        
        # Add confidence threshold
        fig.add_hline(y=0.8, line_dash="dash", line_color="red", 
                     annotation_text="High Confidence Threshold", row=1, col=1)
        
        # Patterns trace
        fig.add_trace(
            go.Scatter(x=x, y=patterns_found, mode='lines+markers',
                      name='Patterns', line=dict(color='#ab47bc', width=3, shape='hv'),
                      marker=dict(size=8, symbol='square')),
            row=2, col=1
        )
        
        fig.update_xaxes(title_text="Evaluation Number", row=2, col=1, gridcolor='rgba(255,255,255,0.1)')
        fig.update_yaxes(title_text="Confidence", row=1, col=1, gridcolor='rgba(255,255,255,0.1)')
        fig.update_yaxes(title_text="Patterns", row=2, col=1, gridcolor='rgba(255,255,255,0.1)')
        
        fig.update_layout(
            title="Level 3: Autonomous Agent - Learning & Pattern Recognition",
            height=600,
            showlegend=True,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0.05)'
        )
        
        return fig
    
    else:  # Level 4
        # Sunburst chart showing agent hierarchy and decision paths
        fig = go.Figure(go.Sunburst(
            labels=['Multi-Agent<br>System', 
                   'Specialists', 'Consensus', 'Arbitration',
                   'Fraud Specialist', 'Risk Analyst', 'Customer Advocate',
                   'Majority Vote', 'Split Decision',
                   'Senior Review', 'Auto-Escalate',
                   'Approve', 'Investigate', 'Deny'],
            parents=['', 
                    'Multi-Agent<br>System', 'Multi-Agent<br>System', 'Multi-Agent<br>System',
                    'Specialists', 'Specialists', 'Specialists',
                    'Consensus', 'Consensus',
                    'Arbitration', 'Arbitration',
                    'Majority Vote', 'Senior Review', 'Senior Review'],
            values=[100, 
                   40, 30, 30,
                   13, 13, 14,
                   20, 10,
                   20, 10,
                   15, 10, 5],
            marker=dict(
                colors=['#37474f',
                       '#546e7a', '#607d8b', '#78909c',
                       '#ef5350', '#42a5f5', '#66bb6a',
                       '#9ccc65', '#ff7043',
                       '#ff6f00', '#ffab40',
                       '#81c784', '#ffd54f', '#ef9a9a'],
                line=dict(color='white', width=2)
            ),
            textinfo='label+percent parent',
            hovertemplate='<b>%{label}</b><br>%{value}% of decisions<br><extra></extra>'
        ))
        
        fig.update_layout(
            title="Level 4: Multi-Agent System - Decision Distribution",
            height=600,
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=12, color='white')
        )
        
        return fig

def create_architecture_diagram(level):
    """Create enhanced architecture diagrams for different agent levels"""
    
    if level == 1:
        # Level 1: Reactive Agent - Simple Decision Tree
        fig = go.Figure()
        
        # Create a sankey diagram for the decision flow
        fig = go.Figure(data=[go.Sankey(
            node = dict(
                pad = 15,
                thickness = 20,
                line = dict(color = "black", width = 0.5),
                label = ["Claim Input", "Rule Engine", "Amount Check", 
                        "High Amount (>$10k)", "Low Amount (≤$10k)", 
                        "Investigate", "Approve"],
                color = ["#64b5f6", "#fff59d", "#ffcc80", 
                        "#ef5350", "#66bb6a", 
                        "#ff7043", "#81c784"],
                x = [0, 0.3, 0.5, 0.7, 0.7, 1, 1],
                y = [0.5, 0.5, 0.5, 0.3, 0.7, 0.3, 0.7]
            ),
            link = dict(
                source = [0, 1, 2, 2, 3, 4],
                target = [1, 2, 3, 4, 5, 6],
                value = [1, 1, 0.3, 0.7, 0.3, 0.7],
                color = ["rgba(100, 181, 246, 0.4)", "rgba(255, 245, 157, 0.4)", 
                        "rgba(239, 83, 80, 0.4)", "rgba(102, 187, 106, 0.4)",
                        "rgba(255, 112, 67, 0.4)", "rgba(129, 199, 132, 0.4)"]
            )
        )])
        
        fig.update_layout(
            title="Level 1: Reactive Agent - Rule-Based Decision Flow",
            font_size=12,
            height=500,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig
    
    elif level == 2:
        # Level 2: Assistant Agent - Network Graph
        import networkx as nx
        
        # Create network graph
        G = nx.Graph()
        
        # Add nodes
        G.add_node("LLM", pos=(0, 0), size=60, color='#ffd54f')
        G.add_node("Check Policy", pos=(-1.5, 1), size=40, color='#90caf9')
        G.add_node("Risk Score", pos=(1.5, 1), size=40, color='#90caf9')
        G.add_node("Verify Docs", pos=(-1.5, -1), size=40, color='#90caf9')
        G.add_node("Fraud Check", pos=(1.5, -1), size=40, color='#90caf9')
        G.add_node("Input", pos=(0, 2), size=30, color='#a5d6a7')
        G.add_node("Output", pos=(0, -2), size=30, color='#ef9a9a')
        
        # Add edges
        edges = [
            ("Input", "LLM"),
            ("LLM", "Check Policy"),
            ("LLM", "Risk Score"),
            ("LLM", "Verify Docs"),
            ("LLM", "Fraud Check"),
            ("LLM", "Output")
        ]
        G.add_edges_from(edges)
        
        pos = nx.get_node_attributes(G, 'pos')
        
        # Create edge traces
        edge_trace = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_trace.append(go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=2, color='#888'),
                hoverinfo='none',
                showlegend=False
            ))
        
        # Create node trace
        node_trace = go.Scatter(
            x=[pos[node][0] for node in G.nodes()],
            y=[pos[node][1] for node in G.nodes()],
            mode='markers+text',
            text=list(G.nodes()),
            textposition="middle center",
            textfont=dict(size=12, color='white'),
            marker=dict(
                size=[G.nodes[node].get('size', 40) for node in G.nodes()],
                color=[G.nodes[node].get('color', '#90caf9') for node in G.nodes()],
                line=dict(width=2, color='white')
            ),
            hoverinfo='text',
            showlegend=False
        )
        
        # Create figure
        fig = go.Figure(data=edge_trace + [node_trace])
        
        # Add annotations for tool types
        annotations = [
            dict(x=-2.2, y=0, text="<b>MCP Tools</b>", showarrow=False, 
                 font=dict(size=14, color='#90caf9')),
            dict(x=0, y=2.5, text="<b>Claim Data</b>", showarrow=False,
                 font=dict(size=12, color='#a5d6a7')),
            dict(x=0, y=-2.5, text="<b>Decision</b>", showarrow=False,
                 font=dict(size=12, color='#ef9a9a'))
        ]
        
        fig.update_layout(
            title="Level 2: Assistant Agent - LLM with MCP Tools",
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=annotations,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=600,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig
    
    elif level == 3:
        # Level 3: Autonomous Agent - Circular Flow with Memory
        fig = go.Figure()
        
        # Create circular layout
        n_nodes = 8
        angles = np.linspace(0, 2*np.pi, n_nodes, endpoint=False)
        radius = 2
        center_x, center_y = 0, 0
        
        nodes = [
            ('Claim Input', '#81c784', 0),
            ('Memory Recall', '#ba68c8', 1),
            ('Pattern Analysis', '#9575cd', 2),
            ('LLM Decision', '#ffd54f', 3),
            ('Confidence Check', '#ff8a65', 4),
            ('Tool Execution', '#4dd0e1', 5),
            ('Memory Update', '#f06292', 6),
            ('Output/Escalate', '#aed581', 7)
        ]
        
        x_pos = center_x + radius * np.cos(angles)
        y_pos = center_y + radius * np.sin(angles)
        
        # Draw circular arrows between nodes
        arrow_traces = []
        for i in range(n_nodes):
            next_i = (i + 1) % n_nodes
            
            # Calculate arrow path (curved)
            t = np.linspace(0, 1, 20)
            x_curve = x_pos[i] * (1-t) + x_pos[next_i] * t
            y_curve = y_pos[i] * (1-t) + y_pos[next_i] * t
            
            # Add curvature
            mid = len(t) // 2
            curve_factor = 0.2
            x_curve[mid] += curve_factor * center_x
            y_curve[mid] += curve_factor * center_y
            
            arrow_traces.append(go.Scatter(
                x=x_curve, y=y_curve,
                mode='lines',
                line=dict(color='rgba(255,255,255,0.3)', width=3),
                hoverinfo='none',
                showlegend=False
            ))
        
        # Add nodes
        node_trace = go.Scatter(
            x=x_pos, y=y_pos,
            mode='markers+text',
            text=[node[0] for node in nodes],
            textposition=[
                "top center", "middle left", "middle left", "bottom center",
                "bottom center", "middle right", "middle right", "top center"
            ],
            textfont=dict(size=12, color='white'),
            marker=dict(
                size=50,
                color=[node[1] for node in nodes],
                line=dict(width=2, color='white')
            ),
            hoverinfo='text',
            showlegend=False
        )
        
        # Add center node
        center_trace = go.Scatter(
            x=[center_x], y=[center_y],
            mode='markers+text',
            text=['Autonomous<br>Agent Core'],
            textposition="middle center",
            textfont=dict(size=14, color='white', family='Arial Black'),
            marker=dict(
                size=80,
                color='#37474f',
                line=dict(width=3, color='white')
            ),
            showlegend=False
        )
        
        # Add memory database
        memory_trace = go.Scatter(
            x=[3], y=[2],
            mode='markers+text',
            text=['Pattern<br>Database'],
            textposition="middle center",
            textfont=dict(size=12, color='white'),
            marker=dict(
                size=60,
                color='#4caf50',
                symbol='square',
                line=dict(width=2, color='white')
            ),
            showlegend=False
        )
        
        # Connection to memory
        memory_connection = go.Scatter(
            x=[x_pos[1], 3, x_pos[2]],
            y=[y_pos[1], 2, y_pos[2]],
            mode='lines',
            line=dict(color='rgba(76, 175, 80, 0.5)', width=4, dash='dash'),
            showlegend=False
        )
        
        fig = go.Figure(data=arrow_traces + [memory_connection, node_trace, center_trace, memory_trace])
        
        fig.update_layout(
            title="Level 3: Autonomous Agent - Self-Directed Learning Loop",
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-3.5, 4]),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-3, 3]),
            height=600,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            annotations=[
                dict(x=3.5, y=2.5, text="<b>Learning & Memory</b>", 
                     showarrow=False, font=dict(size=12, color='#4caf50'))
            ]
        )
        
        return fig
    
    else:  # Level 4
        # Level 4: Multi-Agent System - 3D-like Hierarchy
        fig = go.Figure()
        
        # Define positions for 3D effect
        levels = {
            'coordinator': (0, 3, 100, '#ffd54f'),
            'fraud': (-2, 1.5, 80, '#ef5350'),
            'risk': (0, 1.5, 80, '#42a5f5'),
            'customer': (2, 1.5, 80, '#66bb6a'),
            'consensus': (0, 0, 70, '#ff7043'),
            'arbitrator': (0, -1.5, 90, '#ff6f00'),
            'output': (0, -3, 60, '#9ccc65')
        }
        
        labels = {
            'coordinator': 'Agent<br>Coordinator',
            'fraud': 'Fraud<br>Specialist',
            'risk': 'Risk<br>Analyst',
            'customer': 'Customer<br>Advocate',
            'consensus': 'Consensus<br>Engine',
            'arbitrator': 'Senior<br>Arbitrator',
            'output': 'Final<br>Decision'
        }
        
        # Draw connections with different styles
        connections = [
            ('coordinator', 'fraud', 'solid'),
            ('coordinator', 'risk', 'solid'),
            ('coordinator', 'customer', 'solid'),
            ('fraud', 'consensus', 'solid'),
            ('risk', 'consensus', 'solid'),
            ('customer', 'consensus', 'solid'),
            ('consensus', 'output', 'solid'),
            ('consensus', 'arbitrator', 'dash'),
            ('arbitrator', 'output', 'solid')
        ]
        
        connection_traces = []
        for start, end, style in connections:
            x0, y0, _, _ = levels[start]
            x1, y1, _, _ = levels[end]
            
            color = 'rgba(255, 99, 71, 0.8)' if style == 'dash' else 'rgba(255, 255, 255, 0.4)'
            
            connection_traces.append(go.Scatter(
                x=[x0, x1], y=[y0, y1],
                mode='lines',
                line=dict(color=color, width=3, dash=style),
                hoverinfo='none',
                showlegend=False
            ))
        
        # Add nodes with size based on z-coordinate (depth effect)
        node_trace = go.Scatter(
            x=[pos[0] for pos in levels.values()],
            y=[pos[1] for pos in levels.values()],
            mode='markers+text',
            text=[labels[node] for node in levels.keys()],
            textposition="middle center",
            textfont=dict(
                size=[max(10, pos[2]//8) for pos in levels.values()],
                color='white',
                family='Arial Black'
            ),
            marker=dict(
                size=[pos[2] for pos in levels.values()],
                color=[pos[3] for pos in levels.values()],
                line=dict(width=3, color='white'),
                opacity=0.9
            ),
            hovertext=[f"{labels[node]}<br>Level: {i}" for i, node in enumerate(levels.keys())],
            hoverinfo='text',
            showlegend=False
        )
        
        # Add MCP tools cloud
        tools_trace = go.Scatter(
            x=[-3.5], y=[1.5],
            mode='markers+text',
            text=['MCP<br>Tools'],
            textposition="middle center",
            textfont=dict(size=12, color='white'),
            marker=dict(
                size=60,
                color='#b0bec5',
                symbol='diamond',
                line=dict(width=2, color='white')
            ),
            showlegend=False
        )
        
        # Tool connections
        for agent in ['fraud', 'risk', 'customer']:
            x1, y1, _, _ = levels[agent]
            connection_traces.append(go.Scatter(
                x=[-3.5, x1], y=[1.5, y1],
                mode='lines',
                line=dict(color='rgba(176, 190, 197, 0.3)', width=2, dash='dot'),
                showlegend=False
            ))
        
        fig = go.Figure(data=connection_traces + [node_trace, tools_trace])
        
        # Add annotations
        annotations = [
            dict(x=3, y=2.5, text="<b>Specialist Agents</b>", 
                 showarrow=False, font=dict(size=14, color='#81c784')),
            dict(x=3, y=-1, text="<b>Decision Flow</b>", 
                 showarrow=False, font=dict(size=12, color='#ffab91')),
            dict(x=-3.5, y=0.5, text="Shared<br>Resources", 
                 showarrow=False, font=dict(size=10, color='#b0bec5')),
            dict(x=0, y=-3.7, text="❗ Red dash = No consensus → Arbitration",
                 showarrow=False, font=dict(size=10, color='#ff6b6b'))
        ]
        
        fig.update_layout(
            title={
                'text': "Level 4: Multi-Agent System - Collaborative Intelligence",
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(size=20, color='white')
            },
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-5, 4]),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-4.5, 4]),
            height=700,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            annotations=annotations
        )
        
        return fig

def display_architecture():
    """Enhanced architecture display with visualization options"""
    st.subheader("🏗️ Agent Architecture Deep Dive")
    
    # Visualization style selector
    col1, col2 = st.columns([3, 1])
    with col1:
        selected_level = st.selectbox(
            "Select Agent Level to Explore",
            options=["Level 1: Reactive Agent", "Level 2: Assistant Agent", 
                     "Level 3: Autonomous Agent", "Level 4: Multi-Agent System"]
        )
    
    with col2:
        viz_style = st.radio(
            "Visualization Style",
            ["Network", "Analytics"],
            horizontal=True
        )
    
    st.markdown("---")
    
    # Extract level number
    level_num = int(selected_level.split(":")[0].split()[-1])
    
    # Display the appropriate visualization
    if viz_style == "Network":
        fig = create_architecture_diagram(level_num)
    else:
        fig = create_animated_architecture(level_num)
    
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    # Display architecture details based on selection
    if "Level 1" in selected_level:
        display_level1_architecture()
    elif "Level 2" in selected_level:
        display_level2_architecture()
    elif "Level 3" in selected_level:
        display_level3_architecture()
    elif "Level 4" in selected_level:
        display_level4_architecture()
    
    # Keep the existing expanders for additional info
    with st.expander("🔍 Common Architectural Patterns", expanded=False):
        st.markdown("""
        ### Design Patterns Used
        
        **1. Strategy Pattern**
        - Each agent implements the same evaluation interface
        - Allows swapping agents without changing the system
        
        **2. Observer Pattern**
        - Tool usage logging and callbacks
        - Monitoring agent decisions and actions
        
        **3. Chain of Responsibility**
        - Multi-agent system with escalation
        - Senior arbitration when needed
        
        **4. Memento Pattern**
        - Level 3 agents store state/memory
        - Pattern recognition across evaluations
        
        **5. Facade Pattern**
        - MCP tools provide simple interface to complex operations
        - Hides implementation details from agents
        """)
    
    # MCP Protocol Deep Dive with better visualization
    with st.expander("🔧 MCP Protocol Deep Dive", expanded=False):
        # Create a simple MCP flow diagram
        mcp_fig = go.Figure()
        
        # MCP layers
        layers = ['Tool Definition', 'Registration', 'Execution', 'Integration']
        y_pos = [3, 2, 1, 0]
        colors = ['#e3f2fd', '#bbdefb', '#90caf9', '#64b5f6']
        
        for i, (layer, y, color) in enumerate(zip(layers, y_pos, colors)):
            mcp_fig.add_trace(go.Scatter(
                x=[0, 1, 1, 0, 0],
                y=[y-0.4, y-0.4, y+0.4, y+0.4, y-0.4],
                fill='toself',
                fillcolor=color,
                line=dict(color='white', width=2),
                text=layer,
                mode='lines',
                name=layer,
                hoverinfo='text'
            ))
            
            mcp_fig.add_annotation(
                x=0.5, y=y,
                text=f"<b>{layer}</b>",
                showarrow=False,
                font=dict(size=14, color='black')
            )
        
        mcp_fig.update_layout(
            title="MCP Protocol Stack",
            showlegend=False,
            height=400,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.1, 1.1]),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.5, 3.5]),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(mcp_fig, use_container_width=True)
        
        st.markdown("""
        **Benefits over Function Calling:**
        - Better type safety
        - Clearer separation of concerns
        - Easier testing and debugging
        - Protocol standardization
        """)
    
    # Enhanced performance characteristics
    with st.expander("⚡ Performance Characteristics", expanded=False):
        # Create performance comparison chart
        perf_data = {
            'Metric': ['Latency', 'Token Usage', 'Cost', 'Accuracy'],
            'Level 1': [0.01, 0, 0, 30],
            'Level 2': [3, 1500, 0.01, 70],
            'Level 3': [5, 3000, 0.03, 85],
            'Level 4': [10, 6000, 0.08, 95]
        }
        
        df_perf = pd.DataFrame(perf_data)
        
        fig_perf = go.Figure()
        
        metrics = ['Latency', 'Token Usage', 'Cost', 'Accuracy']
        for i, level in enumerate(['Level 1', 'Level 2', 'Level 3', 'Level 4']):
            values = df_perf[level].tolist()
            # Normalize values for comparison (0-100 scale)
            norm_values = [
                values[0] / 10 * 100,  # Latency (inverted - lower is better)
                values[1] / 60,        # Token usage
                values[2] * 1000,      # Cost
                values[3]              # Accuracy
            ]
            
            fig_perf.add_trace(go.Scatterpolar(
                r=norm_values,
                theta=metrics,
                fill='toself',
                name=level,
                opacity=0.6
            ))
        
        fig_perf.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    gridcolor='rgba(255,255,255,0.2)'
                ),
                bgcolor='rgba(0,0,0,0)',
                angularaxis=dict(gridcolor='rgba(255,255,255,0.2)')
            ),
            showlegend=True,
            title="Performance Comparison Across Agent Levels",
            height=500,
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig_perf, use_container_width=True)
        
        st.markdown("""
        **Optimization Strategies:**
        - Cache common tool results
        - Batch similar evaluations
        - Use Level 1 for initial filtering
        - Escalate only when necessary
        """)



def display_level1_architecture():
    """Display overview of all agent results"""
    results = st.session_state.evaluation_history
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Agents", len(results))
    
    with col2:
        recommendations = [r.recommendation for r in results]
        most_common = max(set(recommendations), key=recommendations.count)
        st.metric("Consensus", most_common.title())
    
    with col3:
        avg_confidence = sum(r.confidence for r in results) / len(results)
        st.metric("Avg Confidence", f"{avg_confidence:.2%}")
    
    with col4:
        avg_time = sum(r.processing_time for r in results) / len(results)
        st.metric("Avg Time", f"{avg_time:.2f}s")
    
    # Comparison chart
    df_results = pd.DataFrame([{
        'Agent': r.evaluator,
        'Recommendation': r.recommendation,
        'Confidence': r.confidence,
        'Risk Level': r.risk_level.value,
        'Processing Time': r.processing_time
    } for r in results])
    
    # Confidence comparison
    fig_confidence = px.bar(
        df_results, 
        x='Agent', 
        y='Confidence',
        title='Agent Confidence Levels',
        color='Recommendation',
        text='Confidence'
    )
    fig_confidence.update_traces(texttemplate='%{text:.2%}', textposition='outside')
    st.plotly_chart(fig_confidence, use_container_width=True)
    
    # Processing time comparison
    fig_time = px.bar(
        df_results,
        x='Agent',
        y='Processing Time',
        title='Processing Time by Agent Level',
        text='Processing Time'
    )
    fig_time.update_traces(texttemplate='%{text:.2f}s', textposition='outside')
    st.plotly_chart(fig_time, use_container_width=True)


def display_agent_details():
    """Display detailed information about each agent"""
    agents = [
        ReactiveInsuranceAgent(),
        AssistantInsuranceAgent(),
        AutonomousInsuranceAgent(),
        MultiAgentInsuranceSystem()
    ]
    
    for i, (agent, result) in enumerate(zip(agents, st.session_state.evaluation_history)):
        with st.expander(f"Level {i+1}: {agent.name}", expanded=i==0):
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown(f"**Description:** {agent.description}")
                st.markdown("**Capabilities:**")
                for cap in agent.capabilities:
                    st.write(cap)
            
            with col2:
                st.markdown("**Evaluation Result:**")
                st.write(f"✅ Recommendation: **{result.recommendation.upper()}**")
                st.write(f"📊 Confidence: **{result.confidence:.2%}**")
                st.write(f"⚠️ Risk Level: **{result.risk_level.value.upper()}**")
                
                if result.reasons:
                    st.markdown("**Reasons:**")
                    for reason in result.reasons:
                        st.write(f"- {reason}")
                
                if result.red_flags:
                    st.markdown("**🚩 Red Flags:**")
                    for flag in result.red_flags:
                        st.write(f"- {flag}")


def display_tool_usage():
    """Display MCP tool usage across agents"""
    st.subheader("🔧 MCP Tool Usage Analysis")
    
    # Collect all tool usage
    all_tools = []
    for result in st.session_state.evaluation_history:
        if hasattr(result, 'tool_usage') and result.tool_usage:
            for tool_call in result.tool_usage:
                all_tools.append({
                    'Agent': result.evaluator,
                    'Tool': tool_call.get('tool', 'Unknown'),
                    'Status': tool_call.get('status', 'Unknown')
                })
    
    if all_tools:
        df_tools = pd.DataFrame(all_tools)
        
        # Tool usage by agent
        fig_tools = px.histogram(
            df_tools,
            x='Agent',
            color='Tool',
            title='Tool Usage by Agent',
            barmode='group'
        )
        st.plotly_chart(fig_tools, use_container_width=True)
        
        # Detailed tool calls
        for agent in df_tools['Agent'].unique():
            agent_tools = df_tools[df_tools['Agent'] == agent]
            if not agent_tools.empty:
                st.markdown(f"**{agent} Tool Calls:**")
                for _, tool in agent_tools.iterrows():
                    st.write(f"- {tool['Tool']} ({tool['Status']})")
    else:
        st.info("No tool usage recorded (Level 1 agent doesn't use tools)")


def display_reasoning():
    """Display agent reasoning and conversations"""
    st.subheader("💭 Agent Reasoning Process")
    
    conversations = st.session_state.agent_conversations
    
    for agent_name, conv_data in conversations.items():
        with st.expander(f"{agent_name} Reasoning", expanded=False):
            if 'thought_process' in conv_data:
                # Level 1: Simple rules
                st.markdown("**Decision Process:**")
                for rule in conv_data['thought_process']:
                    st.write(f"- {rule}")
                
                if 'decision_tree' in conv_data:
                    st.json(conv_data['decision_tree'])
            
            elif 'reasoning' in conv_data:
                # Level 2 & 3: LLM reasoning
                st.markdown("**LLM Reasoning:**")
                if isinstance(conv_data['reasoning'], list):
                    for reasoning in conv_data['reasoning']:
                        st.text_area("Thought Process", reasoning, height=200)
                else:
                    st.text_area("Thought Process", conv_data['reasoning'], height=200)
                
                if 'memory_context' in conv_data:
                    st.markdown("**Memory Context:**")
                    st.text(conv_data['memory_context'])
                
                if 'patterns' in conv_data:
                    st.markdown("**Learned Patterns:**")
                    st.json(conv_data['patterns'])
            
            elif 'agent_conversations' in conv_data:
                # Level 4: Multi-agent
                st.markdown("**Multi-Agent Deliberation:**")
                
                for specialist, spec_conv in conv_data['agent_conversations'].items():
                    st.markdown(f"**{specialist}:**")
                    if 'response' in spec_conv:
                        st.text_area(f"{specialist} Analysis", spec_conv['response'], height=150)
                
                st.markdown(f"**Consensus Reached:** {'✅ Yes' if conv_data.get('consensus') else '❌ No'}")



def display_learning_points():
    """Display educational insights about agent levels"""
    st.subheader("📚 Key Learning Points")
    
    learning_points_dict = {
        "Level 1 - Reactive Agents": [
            "✅ **Pros:** Fast, deterministic, predictable",
            "❌ **Cons:** No context understanding, rigid rules, can't handle edge cases",
            "📝 **Use Case:** Simple, high-volume decisions with clear rules"
        ],
        "Level 2 - Assistant Agents": [
            "✅ **Pros:** LLM reasoning, tool usage, structured analysis",
            "❌ **Cons:** No memory, reactive only, no learning",
            "📝 **Use Case:** Complex analysis requiring reasoning but not continuity"
        ],
        "Level 3 - Autonomous Agents": [
            "✅ **Pros:** Proactive investigation, pattern learning, memory",
            "❌ **Cons:** More complex, requires state management",
            "📝 **Use Case:** Cases requiring historical context and pattern recognition"
        ],
        "Level 4 - Multi-Agent Systems": [
            "✅ **Pros:** Multiple perspectives, specialist knowledge, consensus building",
            "❌ **Cons:** Higher cost, longer processing time, coordination complexity",
            "📝 **Use Case:** High-stakes decisions requiring diverse expertise"
        ]
    }
    
    for agent_level_name, point_list in learning_points_dict.items():
        with st.expander(agent_level_name, expanded=True):
            for point in point_list:
                st.markdown(point)
    
    # MCP Protocol Benefits
    st.markdown("### 🔧 MCP Protocol Benefits")
    st.markdown("""
    - **Structured Tool Calling:** Type-safe, validated inputs and outputs
    - **Separation of Concerns:** Clear distinction between reasoning and tool execution
    - **Scalability:** Same protocol works from simple to complex agents
    - **Observability:** Easy to track and debug tool usage
    - **Extensibility:** Easy to add new tools without changing agent logic
    """)


def display_architecture():
    """Display detailed architecture information for each agent level"""
    st.subheader("🏗️ Agent Architecture Deep Dive")
    
    # Architecture selection
    selected_level = st.selectbox(
        "Select Agent Level to Explore",
        options=["Level 1: Reactive Agent", "Level 2: Assistant Agent", 
                 "Level 3: Autonomous Agent", "Level 4: Multi-Agent System"]
    )
    
    st.markdown("---")
    
    if "Level 1" in selected_level:
        display_level1_architecture()
    elif "Level 2" in selected_level:
        display_level2_architecture()
    elif "Level 3" in selected_level:
        display_level3_architecture()
    elif "Level 4" in selected_level:
        display_level4_architecture()
    
    # Common architectural patterns
    with st.expander("🔍 Common Architectural Patterns", expanded=False):
        st.markdown("""
        ### Design Patterns Used
        
        **1. Strategy Pattern**
        - Each agent implements the same evaluation interface
        - Allows swapping agents without changing the system
        
        **2. Observer Pattern**
        - Tool usage logging and callbacks
        - Monitoring agent decisions and actions
        
        **3. Chain of Responsibility**
        - Multi-agent system with escalation
        - Senior arbitration when needed
        
        **4. Memento Pattern**
        - Level 3 agents store state/memory
        - Pattern recognition across evaluations
        
        **5. Facade Pattern**
        - MCP tools provide simple interface to complex operations
        - Hides implementation details from agents
        """)


def display_level1_architecture():
    """Display Level 1 Reactive Agent architecture"""
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 🎯 Reactive Agent Architecture")
        st.markdown("""
        **Core Components:**
        - **Decision Engine**: Simple if-then-else rules
        - **Input Parser**: Extracts claim amount
        - **Output Generator**: Fixed response templates
        
        **Data Flow:**
        1. Receive claim input
        2. Extract key parameter (amount)
        3. Apply rule threshold
        4. Return fixed response
        """)
        
        # Show example rule implementation
        st.code("""
# Simplified Rule Engine
def evaluate(claim):
    if claim.amount > THRESHOLD:
        return {
            "decision": "investigate",
            "risk": "high"
        }
    else:
        return {
            "decision": "approve",
            "risk": "low"
        }
        """, language="python")
    
    with col2:
        # Architecture diagram using Plotly
        st.markdown("### 📊 Architecture Diagram")
        fig = create_architecture_diagram(1)
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("""
        **Key Characteristics:**
        - No external dependencies
        - Deterministic outcomes
        - Millisecond response times
        - No learning capability
        """)


def display_level2_architecture():
    """Display Level 2 Assistant Agent architecture"""
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 🤖 Assistant Agent Architecture")
        st.markdown("""
        **Core Components:**
        - **LLM Interface**: Claude API integration
        - **MCP Tool Registry**: Available tools catalog
        - **Tool Executor**: Handles tool calls
        - **Response Parser**: Extracts decisions from LLM
        
        **MCP Protocol Flow:**
        1. Register available tools with LLM
        2. LLM decides which tools to use
        3. Execute tools with validated inputs
        4. Return results to LLM
        5. LLM synthesizes final decision
        """)
        
        # Show MCP tool definition
        st.code("""
# MCP Tool Definition
Tool(
    name="calculate_risk_score",
    description="Calculate risk score",
    inputSchema={
        "type": "object",
        "properties": {
            "claim_amount": {"type": "number"},
            "claim_history": {"type": "integer"}
        },
        "required": ["claim_amount"]
    }
)
        """, language="python")
    
    with col2:
        st.markdown("### 📊 Architecture Diagram")
        fig = create_architecture_diagram(2)
        st.plotly_chart(fig, use_container_width=True)
        
        st.success("""
        **MCP Benefits:**
        - Type-safe tool calls
        - Structured inputs/outputs
        - Clear separation of concerns
        - Easy to add new tools
        """)


def display_level3_architecture():
    """Display Level 3 Autonomous Agent architecture"""
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 🧠 Autonomous Agent Architecture")
        st.markdown("""
        **Core Components:**
        - **Memory Store**: Persistent claim history
        - **Pattern Recognizer**: Identifies trends
        - **Proactive Investigator**: Self-directed tools
        - **Confidence Evaluator**: Self-assessment
        - **Escalation Manager**: Senior review trigger
        
        **Advanced Features:**
        1. **Working Memory**: Recent claims context
        2. **Long-term Memory**: Pattern database
        3. **Meta-cognition**: Confidence tracking
        4. **Autonomous Actions**: Self-directed investigation
        """)
        
        # Show memory structure
        st.code("""
# Memory and Pattern Structure
self.memory = [{
    "timestamp": datetime,
    "claim_id": str,
    "type": str,
    "amount": float,
    "outcome": str
}]

self.patterns = {
    "auto_high_risk": {
        "count": 15,
        "outcomes": {
            "approve": 2,
            "deny": 3,
            "investigate": 10
        }
    }
}
        """, language="python")
    
    with col2:
        st.markdown("### 📊 Architecture Diagram")
        fig = create_architecture_diagram(3)
        st.plotly_chart(fig, use_container_width=True)
        
        st.warning("""
        **Autonomous Capabilities:**
        - Self-directed investigation
        - Pattern learning over time
        - Confidence-based escalation
        - Context-aware decisions
        """)


def display_level4_architecture():
    """Display Level 4 Multi-Agent System architecture"""
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 🌐 Multi-Agent System Architecture")
        st.markdown("""
        **Core Components:**
        - **Agent Registry**: Specialist catalog
        - **Communication Bus**: Inter-agent messaging
        - **Consensus Engine**: Decision aggregation
        - **Arbitration System**: Conflict resolution
        - **Coordination Layer**: Workflow management
        
        **Specialist Agents:**
        1. **Fraud Specialist**: Deep fraud analysis
        2. **Risk Analyst**: Comprehensive risk assessment
        3. **Customer Advocate**: Fair treatment focus
        4. **Senior Arbitrator**: Final decisions
        
        **Consensus Mechanisms:**
        - Majority voting
        - Confidence weighting
        - Specialist expertise ranking
        - Escalation protocols
        """)
        
        # Show multi-agent communication
        st.code("""
# Multi-Agent Communication
evaluations = {
    "Fraud Specialist": {
        "recommendation": "investigate",
        "confidence": 0.9,
        "risk": "high"
    },
    "Risk Analyst": {
        "recommendation": "investigate", 
        "confidence": 0.8,
        "risk": "high"
    },
    "Customer Advocate": {
        "recommendation": "approve",
        "confidence": 0.6,
        "risk": "medium"
    }
}

# Consensus: 2/3 vote for "investigate"
# Arbitration: Not needed (majority)
        """, language="python")
    
    with col2:
        st.markdown("### 📊 Architecture Diagram")
        fig = create_architecture_diagram(4)
        st.plotly_chart(fig, use_container_width=True)
        
        st.error("""
        **System Complexity:**
        - Multiple parallel LLM calls
        - Complex coordination logic
        - Higher latency (3-4x)
        - Richer decision context
        """)
    
    # Additional architectural insights
    with st.expander("🔬 Technical Implementation Details"):
        st.markdown("""
        ### Coordination Patterns
        
        **1. Parallel Execution**
        ```python
        # All specialists evaluate simultaneously
        tasks = [
            evaluate_as_fraud_specialist(claim),
            evaluate_as_risk_analyst(claim),
            evaluate_as_customer_advocate(claim)
        ]
        results = await asyncio.gather(*tasks)
        ```
        
        **2. Message Passing**
        - Each agent has isolated context
        - Results aggregated by coordinator
        - No direct agent-to-agent communication
        
        **3. Escalation Protocol**
        - Triggered by: No consensus, Low confidence, High stakes
        - Senior arbitrator has access to all specialist views
        - Final decision binding
        """)


def display_architecture():
    """Display detailed architecture information for each agent level"""
    st.subheader("🏗️ Agent Architecture Deep Dive")
    
    # Architecture selection
    selected_level = st.selectbox(
        "Select Agent Level to Explore",
        options=["Level 1: Reactive Agent", "Level 2: Assistant Agent", 
                 "Level 3: Autonomous Agent", "Level 4: Multi-Agent System"]
    )
    
    st.markdown("---")
    
    if "Level 1" in selected_level:
        display_level1_architecture()
    elif "Level 2" in selected_level:
        display_level2_architecture()
    elif "Level 3" in selected_level:
        display_level3_architecture()
    elif "Level 4" in selected_level:
        display_level4_architecture()
    
    # Common architectural patterns
    with st.expander("🔍 Common Architectural Patterns", expanded=False):
        st.markdown("""
        ### Design Patterns Used
        
        **1. Strategy Pattern**
        - Each agent implements the same evaluation interface
        - Allows swapping agents without changing the system
        
        **2. Observer Pattern**
        - Tool usage logging and callbacks
        - Monitoring agent decisions and actions
        
        **3. Chain of Responsibility**
        - Multi-agent system with escalation
        - Senior arbitration when needed
        
        **4. Memento Pattern**
        - Level 3 agents store state/memory
        - Pattern recognition across evaluations
        
        **5. Facade Pattern**
        - MCP tools provide simple interface to complex operations
        - Hides implementation details from agents
        """)
    
    # MCP Protocol Deep Dive
    with st.expander("🔧 MCP Protocol Deep Dive", expanded=False):
        st.markdown("""
        ### Model Context Protocol (MCP) Architecture
        
        **Protocol Layers:**
        
        1. **Tool Definition Layer**
           - JSON Schema for input validation
           - Strongly typed parameters
           - Clear tool descriptions
        
        2. **Registration Layer**
           - Tools registered with LLM at conversation start
           - LLM understands available capabilities
           - Dynamic tool discovery possible
        
        3. **Execution Layer**
           - LLM generates tool calls
           - Framework validates inputs
           - Tools executed in isolation
           - Results returned to LLM
        
        4. **Integration Layer**
           - LLM combines tool results
           - Reasoning over multiple tool outputs
           - Final decision synthesis
        
        **Benefits over Function Calling:**
        - Better type safety
        - Clearer separation of concerns
        - Easier testing and debugging
        - Protocol standardization
        """)
    
    # Performance Characteristics
    with st.expander("⚡ Performance Characteristics", expanded=False):
        st.markdown("""
        ### Performance Analysis by Level
        
        | Agent Level | Latency | Token Usage | Cost | Accuracy |
        |------------|---------|-------------|------|-----------|
        | Level 1 | ~10ms | 0 | $0 | Low |
        | Level 2 | ~2-3s | ~1.5k | ~$0.01 | Medium |
        | Level 3 | ~4-5s | ~3k | ~$0.03 | High |
        | Level 4 | ~8-10s | ~6k | ~$0.08 | Very High |
        
        **Scaling Considerations:**
        - Level 1: Can handle millions of requests
        - Level 2: Good for thousands per minute
        - Level 3: Stateful, requires memory management
        - Level 4: Best for critical decisions only
        
        **Optimization Strategies:**
        - Cache common tool results
        - Batch similar evaluations
        - Use Level 1 for initial filtering
        - Escalate only when necessary
        """)


if __name__ == "__main__":
    main()