import gradio as gr
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import InMemorySaver
import uuid

from regulatory_agent.multi_agent import MultiAgentState, RegulatoryMultiAgent
from langgraph.graph import END, START, StateGraph


# HITL 노드 (Gradio에서 입력을 처리)
def human_review(state: MultiAgentState):
    pass


class GradioInterface(RegulatoryMultiAgent):
    def _compile_graph(self):
        workflow = StateGraph(MultiAgentState)
        workflow.add_node("route_question", self.route_question)
        workflow.add_node("regulation_agent", self.regulation_node)
        workflow.add_node("guidance_agent", self.guidance_node)
        workflow.add_node("web_agent", self.web_node)
        workflow.add_node("generate_answer", self.generate_answer)
        workflow.add_node("evaluate_answer", self.evaluate_answer)
        workflow.add_node(
            "route_after_evaluation",
            self.route_after_evaluation,
            destinations=(
                "revise_answer",
                "prepare_research_retry",
                "human_review",
                END,
            ),
        )
        workflow.add_node("revise_answer", self.revise_answer)
        workflow.add_node("validate_revision", self.validate_revision)
        workflow.add_node("prepare_research_retry", self.prepare_research_retry)
        workflow.add_node("validate_retry_plan", self.validate_retry_plan)
        workflow.add_node("execute_research_query", self.execute_research_query)
        workflow.add_node("merge_research_evidence", self.merge_research_evidence)
        workflow.add_node("regenerate_answer", self.regenerate_answer)
        workflow.add_node("human_review", self.human_review)
        workflow.add_edge(START, "route_question")
        workflow.add_conditional_edges(
            "route_question",
            self.selected_agent_nodes,
            ["regulation_agent", "guidance_agent", "web_agent"],
        )
        for node in ("regulation_agent", "guidance_agent", "web_agent"):
            workflow.add_edge(node, "generate_answer")
        workflow.add_edge("generate_answer", "evaluate_answer")
        workflow.add_edge("evaluate_answer", "route_after_evaluation")
        workflow.add_edge("revise_answer", "validate_revision")
        workflow.add_edge("validate_revision", "evaluate_answer")
        workflow.add_edge("prepare_research_retry", "validate_retry_plan")
        workflow.add_conditional_edges(
            "validate_retry_plan",
            self.dispatch_research_retry,
            ["execute_research_query", END],
        )
        workflow.add_edge("execute_research_query", "merge_research_evidence")
        workflow.add_edge("merge_research_evidence", "regenerate_answer")
        workflow.add_edge("regenerate_answer", "evaluate_answer")
        workflow.add_conditional_edges(
            "human_review",
            lambda x: "approved" if x.get("user_decision") == "approved" else "rejected",
            {"approved": END, 
             "rejected": "analyzer_question"}
        )
        return workflow.compile(checkpointer=self.checkpointer)