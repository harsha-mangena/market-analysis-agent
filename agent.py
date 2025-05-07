from langchain_google_vertexai import ChatVertexAI
from langchain.agents import initialize_agent, Tool
from data_ingestion import get_stock_data

def init_agent():
    """
    Initializes the LangChain agent with Vertex AI model and tools.
    
    Returns:
        AgentExecutor: Configured LangChain agent
    """
    # Initialize Vertex AI model
    llm = ChatVertexAI(model_name="gemini-2.0-flash")
    
    # Define tools
    tools = [
        Tool(
            name="StockData",
            func=get_stock_data,
            description="Fetches daily stock data for a given symbol. Returns limited data due to free API tier limitations."
        )
    ]
    
    # System message to handle the API limitations
    system_message = """You are a market analysis assistant. Due to API limitations, you might not have access to all stock data or there might be instances where you need to work with limited or simulated data.

Important notes:
1. You only have access to basic daily stock price data through the free Alpha Vantage API
2. You cannot access fundamental data like P/E ratios, EPS, or revenue figures
3. Focus on providing insights based on available price data
4. If asked about fundamental analysis, explain that you have limited data access
5. When generating price targets, be clear that they are estimates based on limited historical data

When analyzing stocks:
- Focus on price trends, support/resistance levels, and basic technical indicators
- Be honest about data limitations
- Provide useful insights that can be made from the available price data

The visualization system will automatically create charts with technical indicators."""
    
    # Initialize LangChain agent with system message
    agent = initialize_agent(
        tools,
        llm,
        agent="zero-shot-react-description",
        verbose=True,
        agent_kwargs={
            "system_message": system_message
        }
    )
    return agent