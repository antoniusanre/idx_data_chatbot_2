import streamlit as st

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain.agents import create_tool_calling_agent, AgentExecutor

import os
import json
import requests
from dotenv import load_dotenv
import datetime
from langchain_core.tools import tool
import time

# GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
# SECTORS_API_KEY = st.secrets["SECTORS_API_KEY"]
# CALENDAR_API_KEY = st.secrets["CALENDAR_API_KEY"]


if "sectors_api_key" not in st.session_state:
    st.session_state["sectors_api_key"] = ""
if "groq_api_key" not in st.session_state:
    st.session_state["groq_api_key"] = ""

with st.sidebar:
    SECTORS_API_KEY = st.text_input(
        "Sectors API Key", key="sectors_api_key", type="password"
    )
    GROQ_API_KEY = st.text_input("Groq API Key", key="groq_api_key", type="password")
    button = st.button("Set API Keys")

    if button:
        st.write("API Keys set!")

    st.link_button("Get Sectors API Key", "https://sectors.app/api")
    st.link_button("Get Groq API Key", "https://console.groq.com/keys")

def get_today_date() -> str:
    """
    Get the date for today.
    """
    today = datetime.date.today()
    return today.strftime("%Y-%m-%d")

def retrieve_from_endpoint(url: str) -> dict:
    headers = {"Authorization": SECTORS_API_KEY}

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        # If error
        data = f"Currently there are no content at this time. Try again in other time"
        if response is not None and response.text != '':
            data = response.json()
    except requests.exceptions.HTTPError as err:
        raise SystemExit(err)

    return data

def fetch_holidays_and_mass_leave(year: int) -> set:
    cache_file = f"holidays_{year}.json"
    
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            holiday_dates = json.load(f)
    else:
        url = f"https://calendarific.com/api/v2/holidays?&api_key={CALENDAR_API_KEY}&country=ID&year={year}"
        response = requests.get(url)
        print(response)
        holidays = response.json().get('response', {}).get('holidays', [])
        print(holidays)
        holiday_dates = {holiday['date']['iso'] for holiday in holidays if holiday['type'][0] in ['National holiday', 'Public holiday','Observance']}
        print(holiday_dates)
        with open(cache_file, 'w') as f:
            json.dump([str(date) for date in holiday_dates], f)
    
    return holiday_dates

def is_weekend(date: str) -> bool:
    date = datetime.datetime.strptime(date, '%Y-%m-%d').date()
    return date.isoweekday() > 5

def is_holiday(date:str) -> bool:
    date = datetime.datetime.strptime(date, '%Y-%m-%d').date()
    holiday_date = fetch_holidays_and_mass_leave(date.year)
    return str(date) in holiday_date

def check_work_day(date) -> bool:
    return not (is_weekend(date) or is_holiday(date))

def closest_working_day(date:str, method = 'backward') -> str:
    temp_date = datetime.datetime.strptime(date, '%Y-%m-%d').date()
    if method == 'backward':
        while not check_work_day(str(temp_date)):
            temp_date = temp_date - datetime.timedelta(1)
    if method == 'forward':
        while not check_work_day(str(temp_date)):
            temp_date = temp_date + datetime.timedelta(1)
    return str(temp_date)

def date_handler(start_date:str, end_date:str):
    """
    handle the start and end date.
    """
    temp_start_date = datetime.datetime.strptime(closest_working_day(start_date, 'forward'), '%Y-%m-%d').date()
    temp_end_date = datetime.datetime.strptime(closest_working_day(end_date, 'backward'), '%Y-%m-%d').date()
    if (temp_end_date - temp_start_date).days < 0:
        temp_start_date = temp_end_date
    return str(temp_start_date), str(temp_end_date)

def get_company_list() -> str:
    """
    Get list of company in sectors app.
    """
    sector = []
    sub_sector = []
    company_list = []

    if  not os.path.isfile('./company.json'):
        raw_subsector = json.loads(retrieve_from_endpoint(f"https://api.sectors.app/v1/subsectors/"))
        print(raw_subsector)
        for x in raw_subsector:
            print(x)
            if x['sector'] not in sector:
                sector.append(x['sector'])
            sub_sector.append(x['subsector'])
        for x in sub_sector:
            raw_company = json.loads(retrieve_from_endpoint(f"https://api.sectors.app/v1/companies/?sub_sector={x}"))
            company_list.append(raw_company)
            time.sleep(5)

        with open('sector.txt', 'w') as f:
            for s in sector:
                f.write(f"{s}\n")
        with open('subsector.txt', 'w') as f:
            for sb in sub_sector:
                f.write(f"{sb}\n")
        with open('company.json', 'w') as f:
            for c in company_list:
                f.write(f"{c}\n")

    with open(r"company.json") as file:
        company_list = json.load(file)
    with open(r"sector.txt") as file:
        for line in file:
            line = line.strip()
            sector.append(line)
    with open(r"subsector.txt") as file:
        for line in file:
            line = line.strip()
            sub_sector.append(line)

    return sector,sub_sector,company_list

@tool
def get_subsector() -> str:
    """
    Get all the subsector that is listed.
    """
    url = f"https://api.sectors.app/v1/subsectors/"

    return retrieve_from_endpoint(url)

@tool
def get_company_overview(stock: str) -> str:
    """
    Get company or stock overview.
    input: stock/symbol in 4 digit
    output:
        "listing_board"
        "industry
        "sub_industry
        "sector": "Financials
        "sub_sector
        "market_cap"
        "market_cap_rank"
        "address"
        "employee_num"
        "listing_date"
        "website"
        "phone"
        "email"
        "last_close_price"
        "latest_close_date"
        "daily_close_change"
    """
    url = f"https://api.sectors.app/v1/company/report/{stock}/?sections=overview"

    return retrieve_from_endpoint(url)

@tool
def get_top_companies_by_tx_volume(
    start_date: str, end_date: str, top_n: int = 1
) -> str:
    """
    Get top n companies by transaction volume. You can see top companies by transaction volume in IDX. You can also see the most traded stock using this tools.
    input: 
    - start_date -> start of the date range
    - end_date -> end of the date range
    - top_n -> number of top companies
    output:each day of top companies by transaction volume
    - symbol -> company stock symbol
    - company name -> company name
    - volume -> transaction volume on specific date
    - price -> stock price on specific date
    """
    message = []
    new_start_date, new_end_date = date_handler(start_date,end_date)
    if new_start_date != start_date or new_end_date != end_date:
        message.append(f"You can't answer given date because of either the given date is on weekend or holiday, tell human you're sorry. Now the data show for {new_start_date} - {new_end_date}")
    
    url = f"https://api.sectors.app/v1/most-traded/?start={new_start_date}&end={new_end_date}&n_stock={top_n}"
    api_result = retrieve_from_endpoint(url)
    result = api_result

    if new_start_date != new_end_date:
        temp_result = {}
        for date in result:
            for c in result[date]:
                symbol = c['symbol']
                company_name = c['company_name']
                volume = c['volume']
                price = c['price']
                if symbol not in temp_result:
                    temp_result[symbol] = {'company_name': company_name, 'total_volume':0, 'average_price':0, 'top_occurence':0}
                temp_result[symbol]['total_volume'] += volume
                temp_result[symbol]['average_price'] += price
                temp_result[symbol]['top_occurence'] += 1
        
        temp_result = dict(sorted(temp_result.items(), key=lambda item: item[1]['total_volume'], reverse=True))
        temp_result = dict(list(temp_result.items())[:top_n])
        rank = 1
        for c in temp_result:
            temp_result[c]['average_price'] /= temp_result[c]['top_occurence']
            temp_result[c]['total_volume_rank'] = rank
            rank +=1
        result = temp_result

    return [message, result] if message != [] else result


@tool
def get_top_companies_by_growth(
    classifications: str, sub_sector: str, top_n: int = 1
) -> str:
    """
    Get top companies by companies growth. Classifications is one of top_earnings_growth_gainers, top_earnings_growth_losers, top_revenue_growth_gainers, top_revenue_growth_losers. For classification use top_earnings_growth_gainers as default. For sub_sector use Banks as default.
    """
    url = f"https://api.sectors.app/v1/companies/top-growth/?classifications={classifications}&n_stock={top_n}&sub_sector={sub_sector}"

    return retrieve_from_endpoint(url)

@tool
def get_daily_tx(stock: str, start_date: str, end_date: str) -> str:
    """
    Return daily transaction data of a given stock on a certain interval (up to 90 days). It will give you closing price, transaction volume, and also market cap.
    input: stock/symbol in 4 digit, start date, and end date
    """
    url = f"https://api.sectors.app/v1/daily/{stock}/?start={start_date}&end={end_date}"

    return retrieve_from_endpoint(url)

@tool
def get_company_performance_IPO(stock: str) -> str:
    """
    Get listing performance of a stock since IPO. Listing performance data is accessible only for stocks listed after May 2005.
    """
    url = f"https://api.sectors.app/v1/listing-performance/{stock}/"

    return retrieve_from_endpoint(url)

@tool
def get_top_companies_ranked( year:str, top_n :int = 1) -> str:
    """
    Get top companies ranked either by market cap, dividend yield, earnings, revenue, and total dividend.
    """
    url = f"https://api.sectors.app/v1/companies/top/?n_stock={top_n}&year={year}"

    return retrieve_from_endpoint(url)


tools = [
    get_company_overview,
    get_top_companies_by_tx_volume,
    get_daily_tx,
    get_top_companies_by_growth,
    get_subsector,
    get_company_performance_IPO,
    get_top_companies_ranked
]

sector, subsector, company_list = get_company_list()
symbol_list = li = [item.get('symbol') for item in company_list]
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            f"""
            You are intellectual AI that have knowledge on IDX data, your name is IDA (IDX Data Assitant). Answer all the questions according to the tools provided.
            Use today date: {get_today_date()} as time reference. Only use sector, subsector that's listed. List of sector: {sector}. List of subsector: {subsector}.
            If it's the most query return only one company. Answer politely and in pretty format.
            If two different tools were used, use the first tool and then, use the output to second tool get final result, and so on.
            You can't answer data outside IDX or Indonesia Stock Exchange).
            """
        ),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)
llm = ChatGroq(
temperature=0,
model_name="llama3-groq-70b-8192-tool-use-preview",
groq_api_key=GROQ_API_KEY)

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True,  handle_parsing_errors=True)

st.title("IDA (IDX Data Assitant)")

def generate_response(input_text):
    result = agent_executor.invoke({"input": input_text})
    st.info(result["output"])
    return result['output']

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

response = None
# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # check api key
    if (st.session_state["sectors_api_key"] == "" or st.session_state["groq_api_key"] == ""):
        st.error("Please set your API Keys first before using the chatbot.")
        st.stop()
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        stream = generate_response(prompt)
        response = stream
    st.session_state.messages.append({"role": "assistant", "content": response})

def test_query():
    query_1 = "What are the top 3 companies by transaction volume over the last 7 days?"
    query_2 = "Based on the closing prices of BBCA between 1st and 30th of June 2024, are we seeing an uptrend or downtrend? Try to explain why."
    query_3 = "What is the company with the largest market cap between BBCA and BREN? For said company, retrieve the email, phone number, listing date and website for further research."
    query_4 = "What is the performance of GOTO (symbol: GOTO) since its IPO listing?"
    query_5 = "If i had invested into GOTO vs BREN on their respective IPO listing date, which one would have given me a better return over a 90 day horizon?"
    query_6 = "What are the top 5 companies by transaction volume on the first of this month?"
    query_7 = "What are the most traded stock yesterday?"
    query_8 = "What are the top 7 most traded stocks between 6th June to 10th June this year?"
    query_9 = "Top 5 stocks by market cap in IDX in 5-9-2024"
    query_10 = "Top 5 stocks by market cap in Singapore Exchanges in 5-9-2024"

    # queries = [query_1, query_2, query_3, query_4, query_5, query_6, query_7, query_8]
    queries = [query_2,query_3,query_6,query_7, query_8]

    for query in queries:
        print("Question:", query)
        result = agent_executor.invoke({"input": query})
        print("Answer:", "\n", result["output"], "\n\n======\n\n")
