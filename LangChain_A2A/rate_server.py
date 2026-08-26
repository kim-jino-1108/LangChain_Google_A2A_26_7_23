# rate_server.py
# A2A Protocol v1.0 (a2a-sdk 1.x) 서버 + LangChain 에이전트 연결

import uvicorn
from starlette.applications import Starlette

from a2a.helpers import new_text_message
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentInterface, AgentSkill
from a2a.utils.errors import UnsupportedOperationError

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

# .env 파일의 환경변수(OPENAI_API_KEY 등)를 프로세스 환경에 로드
load_dotenv()


@tool
def get_exchange_rate(currency_from: str, currency_to: str) -> str:
    """두 통화 간 환율을 조회한다."""
    # 실제 환율 API 대신 고정값을 반환하는 더미 구현
    return f"1 {currency_from} = 1350 {currency_to}"


# LangChain 에이전트 생성
# checkpointer를 지정하면 thread_id 단위로 메시지 히스토리가 저장되어 멀티턴 대화가 유지됨
lc_agent = create_agent(
    model="openai:gpt-5-mini",
    tools=[get_exchange_rate],
    system_prompt="너는 환율 전문 어시스턴트다.",
    checkpointer=InMemorySaver(),
)


class CurrencyAgentExecutor(AgentExecutor):
    """A2A 프로토콜 계층과 LangChain 에이전트를 연결하는 어댑터"""

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        try:
            # context.get_user_input(): 요청 메시지의 모든 text Part를 하나의 문자열로 합쳐서 반환
            # context.context_id: A2A의 대화 식별자. 클라이언트가 보내지 않으면 서버가 자동 생성함
            # 이 값을 LangGraph thread_id로 넘겨 같은 대화의 히스토리를 이어감
            result = await lc_agent.ainvoke(
                {"messages": [{"role": "user", "content": context.get_user_input()}]},
                config={"configurable": {"thread_id": context.context_id}},
            )
            # BaseMessage.text: content가 문자열이든 컨텐츠 블록 리스트이든
            # text 타입만 뽑아 문자열로 정규화해주는 langchain-core 1.x의 property
            answer = result["messages"][-1].text

        except Exception as e:
            # 예외를 그대로 올리면 JSON-RPC 에러 응답이 되므로,
            # 정상 응답 메시지 본문에 오류 내용을 담아 클라이언트가 읽을 수 있게 함
            answer = f"에이전트 실행 중 오류가 발생했습니다: {type(e).__name__}: {e}"

        # v1.0의 message-only 스트림 규칙: Message를 정확히 한 번만 enqueue하고 종료
        # new_text_message는 role 기본값이 ROLE_AGENT이므로 별도 지정 불필요
        await event_queue.enqueue_event(new_text_message(answer))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        # 작업 취소를 지원하지 않으므로 A2A 표준 에러로 응답
        raise UnsupportedOperationError()


# 에이전트 카드: 다른 에이전트가 이 서버의 능력을 발견하는 데 쓰는 메타데이터
agent_card = AgentCard(
    name="Currency Agent",
    description="LangChain 기반 환율 에이전트",
    version="1.0.0",
    # v1.0에서 AgentCard.url이 제거되고 supported_interfaces로 대체됨
    # protocol_binding에는 'JSONRPC' / 'HTTP+JSON' / 'GRPC'를 지정
    supported_interfaces=[
        AgentInterface(
            url="http://localhost:9999/",
            protocol_binding="JSONRPC",
            protocol_version="1.0",
        )
    ],
    default_input_modes=["text"],
    default_output_modes=["text"],
    # 실제 구현이 중간 이벤트를 발행하지 않으므로 streaming=False로 선언
    capabilities=AgentCapabilities(streaming=False, push_notifications=False),
    skills=[
        AgentSkill(
            id="currency_convert",
            name="Currency Conversion",
            description="환율 조회 및 통화 변환을 수행한다",
            tags=["finance", "currency"],
            examples=["1달러는 몇 원이야?"],
        )
    ],
)

# 요청 핸들러: 프로토콜 요청을 받아 executor를 호출하고 Task 상태를 저장
# v1.0부터 agent_card가 필수 인자
request_handler = DefaultRequestHandler(
    agent_executor=CurrencyAgentExecutor(),
    task_store=InMemoryTaskStore(),
    agent_card=agent_card,
)

# v1.0에서 A2AStarletteApplication이 제거되고 라우트 팩토리 방식으로 변경됨
# create_agent_card_routes: GET /.well-known/agent-card.json (card_url 기본값)
# create_jsonrpc_routes: POST / (rpc_url은 기본값이 없어 반드시 지정)
app = Starlette(
    routes=create_agent_card_routes(agent_card)
    + create_jsonrpc_routes(request_handler, "/")
)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9999)