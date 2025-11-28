# backend/orchestration/orchestrator.py
"""
Enhanced Orchestrator with LangGraph

Implements multi-agent workflow with:
- State machine for conversation flows
- Dynamic routing based on intent
- Parallel and sequential execution
- Memory integration
- Self-reflection and quality control
"""

from typing import Dict, Any, List, Optional, TypedDict, Annotated
from datetime import datetime
import asyncio
from loguru import logger

# LangGraph imports
try:
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import MemorySaver
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    logger.warning("LangGraph not available, using simplified orchestration")

# Local imports
from backend.agents import (
    QueryClassifier, QueryIntent,
    ClarificationAgent, AnalysisAgent,
    SummarizationAgent, ExplanationAgent,
    CriticAgent
)
from backend.memory import MemoryManager


class WorkflowState(TypedDict):
    """State for the workflow graph"""
    # Input
    user_id: str
    session_id: str
    query: str

    # Classification
    intent: Optional[str]
    confidence: float
    filters: Optional[Dict]
    entities: List[str]

    # Search results
    results: List[Dict[str, Any]]
    search_time: float

    # Agent outputs
    clarification_questions: List[str]
    comparison_result: Optional[Dict]
    summary: Optional[str]
    explanations: List[str]
    insights: List[str]

    # Quality control
    quality_evaluation: Optional[Dict]
    should_reformulate: bool

    # Response
    response_message: str
    suggestions: List[str]

    # Context
    session_context: Optional[Dict]
    user_preferences: Optional[Dict]

    # Control flow
    next_action: str
    error: Optional[str]


class EnhancedOrchestrator:
    """
    Enhanced orchestrator using LangGraph for stateful multi-agent workflows

    Workflow:
    1. Classify query → Route based on intent
    2. For document queries: Search → Quality check → Explain
    3. For general queries: Answer directly
    4. For ambiguous: Clarify → Refine → Search
    5. For analysis: Search → Analyze → Summarize
    """

    def __init__(
        self,
        config: Dict[str, Any],
        memory_manager: MemoryManager,
        search_function: callable
    ):
        self.config = config
        self.memory = memory_manager
        self.search_function = search_function

        # Initialize agents
        self.classifier = QueryClassifier(config)
        self.clarification_agent = ClarificationAgent(config)
        self.analysis_agent = AnalysisAgent(config)
        self.summarization_agent = SummarizationAgent(config)
        self.explanation_agent = ExplanationAgent(config)
        self.critic_agent = CriticAgent(config)

        # Build workflow graph
        if LANGGRAPH_AVAILABLE:
            self.workflow = self._build_langgraph_workflow()
        else:
            self.workflow = None

        logger.info("✅ Enhanced Orchestrator initialized")

    def _build_langgraph_workflow(self) -> StateGraph:
        """
        Build LangGraph workflow

        Graph structure:
        START → Classify → [Route based on intent] → Process → QualityCheck → END
        """
        workflow = StateGraph(WorkflowState)

        # Add nodes
        workflow.add_node("classify", self._classify_node)
        workflow.add_node("load_context", self._load_context_node)
        workflow.add_node("document_search", self._document_search_node)
        workflow.add_node("general_answer", self._general_answer_node)
        workflow.add_node("clarify", self._clarify_node)
        workflow.add_node("analyze", self._analyze_node)
        workflow.add_node("summarize", self._summarize_node)
        workflow.add_node("explain", self._explain_node)
        workflow.add_node("quality_check", self._quality_check_node)
        workflow.add_node("generate_response", self._generate_response_node)

        # Set entry point
        workflow.set_entry_point("load_context")

        # Add edges
        workflow.add_edge("load_context", "classify")

        # Conditional routing based on intent
        workflow.add_conditional_edges(
            "classify",
            self._route_by_intent,
            {
                "document_search": "document_search",
                "general_knowledge": "general_answer",
                "clarification": "clarify",
                "comparison": "analyze",
                "summarization": "summarize",
                "analysis": "analyze"
            }
        )

        # Search → Explain → Quality Check
        workflow.add_edge("document_search", "explain")
        workflow.add_edge("explain", "quality_check")

        # Analysis → Quality Check
        workflow.add_edge("analyze", "quality_check")

        # Summarize → Quality Check
        workflow.add_edge("summarize", "quality_check")

        # General answer → Quality Check
        workflow.add_edge("general_answer", "quality_check")

        # Clarify → END (need user input)
        workflow.add_edge("clarify", "generate_response")

        # Quality Check → Response
        workflow.add_edge("quality_check", "generate_response")

        # Response → END
        workflow.add_edge("generate_response", END)

        # Compile
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)

    async def process_query(
        self,
        user_id: str,
        session_id: str,
        query: str
    ) -> Dict[str, Any]:
        """
        Process query through orchestrated workflow

        Args:
            user_id: User identifier
            session_id: Session identifier
            query: User query

        Returns:
            Orchestrated response with results and metadata
        """
        start_time = asyncio.get_event_loop().time()

        # Initialize state
        initial_state: WorkflowState = {
            "user_id": user_id,
            "session_id": session_id,
            "query": query,
            "intent": None,
            "confidence": 0.0,
            "filters": None,
            "entities": [],
            "results": [],
            "search_time": 0.0,
            "clarification_questions": [],
            "comparison_result": None,
            "summary": None,
            "explanations": [],
            "insights": [],
            "quality_evaluation": None,
            "should_reformulate": False,
            "response_message": "",
            "suggestions": [],
            "session_context": None,
            "user_preferences": None,
            "next_action": "classify",
            "error": None
        }

        try:
            if LANGGRAPH_AVAILABLE and self.workflow:
                # Use LangGraph workflow
                config = {"configurable": {"thread_id": session_id}}
                result = await self.workflow.ainvoke(initial_state, config)
            else:
                # Fallback to simple sequential processing
                result = await self._simple_workflow(initial_state)

            # Record interaction in memory
            await self.memory.record_interaction(
                user_id=user_id,
                session_id=session_id,
                query=query,
                response=result["response_message"],
                results=result["results"],
                intent=result.get("intent", "general"),
                search_time=result["search_time"],
                metadata={
                    "quality_score": result.get("quality_evaluation", {}).get("quality_score"),
                    "confidence": result.get("confidence")
                }
            )

            total_time = asyncio.get_event_loop().time() - start_time
            result["total_time"] = round(total_time, 2)

            return result

        except Exception as e:
            logger.error(f"Orchestration error: {e}")
            return {
                "status": "error",
                "error": str(e),
                "message": "I encountered an error processing your request. Please try again."
            }

    async def _simple_workflow(self, state: WorkflowState) -> WorkflowState:
        """Simplified workflow without LangGraph"""
        # Load context
        state = await self._load_context_node(state)

        # Classify
        state = await self._classify_node(state)

        # Route by intent
        intent = state.get("intent", "document_search")

        if intent == QueryIntent.CLARIFICATION_NEEDED.value:
            state = await self._clarify_node(state)
        elif intent in [QueryIntent.GENERAL_KNOWLEDGE.value, QueryIntent.SYSTEM_META.value]:
            state = await self._general_answer_node(state)
        elif intent in [QueryIntent.COMPARISON.value, QueryIntent.ANALYSIS.value]:
            state = await self._document_search_node(state)
            state = await self._analyze_node(state)
        elif intent == QueryIntent.SUMMARIZATION.value:
            state = await self._document_search_node(state)
            state = await self._summarize_node(state)
        else:
            # Document search
            state = await self._document_search_node(state)
            state = await self._explain_node(state)

        # Quality check
        if state["results"]:
            state = await self._quality_check_node(state)

        # Generate response
        state = await self._generate_response_node(state)

        return state

    # ===== WORKFLOW NODES =====

    async def _load_context_node(self, state: WorkflowState) -> WorkflowState:
        """Load session context and user preferences"""
        try:
            # Get session context
            context = await self.memory.get_context(state["session_id"])
            state["session_context"] = context

            # Get user preferences
            prefs = await self.memory.get_user_preferences(state["user_id"])
            state["user_preferences"] = prefs

            logger.debug(f"Loaded context for session {state['session_id']}")

        except Exception as e:
            logger.error(f"Context loading failed: {e}")

        return state

    async def _classify_node(self, state: WorkflowState) -> WorkflowState:
        """Classify query intent"""
        try:
            result = await self.classifier.classify(
                query=state["query"],
                context=state.get("session_context")
            )

            state["intent"] = result.intent.value
            state["confidence"] = result.confidence
            state["filters"] = result.filters
            state["entities"] = result.entities
            state["clarification_questions"] = result.clarification_questions

            logger.info(f"Query classified as: {result.intent.value} (confidence: {result.confidence:.2f})")

        except Exception as e:
            logger.error(f"Classification failed: {e}")
            state["intent"] = QueryIntent.DOCUMENT_SEARCH.value

        return state

    async def _document_search_node(self, state: WorkflowState) -> WorkflowState:
        """Perform document search"""
        try:
            import time
            search_start = time.time()

            # Get personalized search config
            prefs = state.get("user_preferences", {})
            optimal_weights = prefs.get("optimal_weights", {})

            # Call search function
            results = await self.search_function(
                query=state["query"],
                filters=state.get("filters"),
                weights=optimal_weights
            )

            state["results"] = results
            state["search_time"] = round(time.time() - search_start, 3)

            logger.info(f"Found {len(results)} results in {state['search_time']:.2f}s")

        except Exception as e:
            logger.error(f"Search failed: {e}")
            state["error"] = str(e)

        return state

    async def _general_answer_node(self, state: WorkflowState) -> WorkflowState:
        """Answer general knowledge questions"""
        try:
            # Use Ollama to answer directly
            import httpx

            prompt = f"""Answer this question concisely:

Question: {state['query']}

Provide a helpful, accurate answer."""

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.config['ollama']['base_url']}/api/generate",
                    json={
                        "model": self.config['ollama']['model'],
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": 0.4}
                    }
                )
                result = response.json()
                answer = result.get('response', '').strip()

            state["response_message"] = answer
            state["results"] = []

        except Exception as e:
            logger.error(f"General answer failed: {e}")
            state["response_message"] = "I couldn't generate an answer. Please try rephrasing your question."

        return state

    async def _clarify_node(self, state: WorkflowState) -> WorkflowState:
        """Handle ambiguous queries"""
        try:
            questions = await self.clarification_agent.generate_clarifying_questions(
                query=state["query"],
                ambiguity_info={
                    "issues": ["ambiguous intent"],
                    "possible_interpretations": []
                },
                max_questions=3
            )

            state["clarification_questions"] = questions
            state["response_message"] = "I need some clarification to help you better:"

        except Exception as e:
            logger.error(f"Clarification failed: {e}")

        return state

    async def _analyze_node(self, state: WorkflowState) -> WorkflowState:
        """Perform cross-document analysis"""
        try:
            results = state.get("results", [])

            if len(results) >= 2:
                comparison = await self.analysis_agent.compare_documents(
                    documents=results[:3]
                )
                state["comparison_result"] = comparison

                # Generate insights
                insights = await self.analysis_agent.generate_insights(
                    documents=results,
                    query=state["query"]
                )
                state["insights"] = insights

        except Exception as e:
            logger.error(f"Analysis failed: {e}")

        return state

    async def _summarize_node(self, state: WorkflowState) -> WorkflowState:
        """Summarize multiple documents"""
        try:
            results = state.get("results", [])

            if results:
                summary = await self.summarization_agent.summarize_documents(
                    documents=results,
                    summary_type="comprehensive"
                )
                state["summary"] = summary

        except Exception as e:
            logger.error(f"Summarization failed: {e}")

        return state

    async def _explain_node(self, state: WorkflowState) -> WorkflowState:
        """Explain search results"""
        try:
            results = state.get("results", [])
            explanations = []

            # Explain top 3 results
            for i, doc in enumerate(results[:3], 1):
                explanation = await self.explanation_agent.explain_ranking(
                    query=state["query"],
                    document=doc,
                    rank=i
                )
                explanations.append(explanation)

            state["explanations"] = explanations

        except Exception as e:
            logger.error(f"Explanation failed: {e}")

        return state

    async def _quality_check_node(self, state: WorkflowState) -> WorkflowState:
        """Perform quality control"""
        try:
            evaluation = await self.critic_agent.evaluate_results(
                query=state["query"],
                results=state.get("results", [])
            )

            state["quality_evaluation"] = evaluation
            state["should_reformulate"] = evaluation.get("should_reformulate", False)

            # Get suggestions
            suggestions = await self.critic_agent.suggest_improvements(
                query=state["query"],
                results=state.get("results", []),
                evaluation=evaluation
            )
            state["suggestions"] = suggestions

        except Exception as e:
            logger.error(f"Quality check failed: {e}")

        return state

    async def _generate_response_node(self, state: WorkflowState) -> WorkflowState:
        """Generate final response message"""
        try:
            intent = state.get("intent", "general")
            results = state.get("results", [])

            # If clarification needed
            if state.get("clarification_questions"):
                state["response_message"] = "I need some clarification:\n" + "\n".join(
                    f"• {q}" for q in state["clarification_questions"]
                )
                return state

            # If general answer already generated
            if state.get("response_message"):
                return state

            # If summarization
            if state.get("summary"):
                state["response_message"] = f"**Summary of {len(results)} documents:**\n\n{state['summary']}"
                return state

            # If comparison
            if state.get("comparison_result"):
                comp = state["comparison_result"]
                msg = f"**Comparison of documents:**\n\n"
                msg += f"**Similarities:** {', '.join(comp.get('similarities', []))}\n"
                msg += f"**Differences:** {', '.join(comp.get('differences', []))}"
                state["response_message"] = msg
                return state

            # Standard document search results
            if results:
                count = len(results)
                state["response_message"] = f"I found {count} relevant document{'s' if count != 1 else ''} for your query."
            else:
                state["response_message"] = f"I couldn't find any documents matching '{state['query']}'. Try different keywords."

        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            state["response_message"] = "I found some results for your query."

        return state

    # ===== ROUTING FUNCTIONS =====

    def _route_by_intent(self, state: WorkflowState) -> str:
        """Route to appropriate node based on intent"""
        intent = state.get("intent", "document_search")
        confidence = state.get("confidence", 1.0)

        # If very low confidence, clarify
        if confidence < 0.3:
            return "clarification"

        # Route by intent
        if intent == QueryIntent.DOCUMENT_SEARCH.value:
            return "document_search"
        elif intent in [QueryIntent.GENERAL_KNOWLEDGE.value, QueryIntent.SYSTEM_META.value]:
            return "general_knowledge"
        elif intent == QueryIntent.CLARIFICATION_NEEDED.value:
            return "clarification"
        elif intent == QueryIntent.COMPARISON.value:
            return "comparison"
        elif intent == QueryIntent.SUMMARIZATION.value:
            return "summarization"
        elif intent == QueryIntent.ANALYSIS.value:
            return "analysis"
        else:
            return "document_search"
