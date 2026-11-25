import json
import random
import altair as alt
import pandas as pd
from datetime import datetime
import httpagentparser
import itertools

from myapp.search.algorithms import build_terms


class AnalyticsData:
    """
    An in memory persistence object.
    Declare more variables to hold analytics tables.
    """
    # Example of statistics table
    # fact_clicks is a dictionary with the click counters: key = doc id | value = click counter
    fact_clicks = dict([])

    ### Please add your custom tables here:
    # start schema, stored in memory as list of dicts
    # dim table: one row per user session
    sessions = [] # {session_id, start_time, ip, browser, os, device}

    #fact table: one row per HTTP request (page visited)
    requests = [] # {session_id, timestamp, path, method}

    #fact table: one row per search query
    queries = [] # {search_id, session_id, timestamp, query, n_terms, n_results, order_in_session}

    #fact table: one row per click on a search result
    clicks = [] # {click_id, session_id, search_id, doc_id, rank, timestamp, dwell_time}

    #counter to give each search a unique id (1,2,3,..)
    _query_counter = itertools.count(1)

    #counter to give each click a unique id (1,2,3,..)
    _click_counter = itertools.count(1)


    def session_exists(self, session_id: str) -> bool:
        """ Return True if the session is already stored in the session table"""
        return any(s["session_id"] == session_id for s in self.sessions)

    def register_session(self, session_id: str, ip: str, user_agent: str):
        """Store a new session (one row per user/browser) with its user context:
        IP, browser, operating system and device type, parsed from the User-Agent"""

        user_agent = user_agent or ""
        agent = httpagentparser.detect(user_agent)
        browser = agent.get("browser", {}).get("name", "Unknown")
        os_name = agent.get("os", {}).get("name", "Unknown")

        ua = user_agent.lower()
        if "ipad" in ua or "tablet" in ua:
            device = "Tablet"
        elif "iphone" in ua or "android" in ua or "mobile" in ua:
            device = "Mobile"
        else:
            device = "Computer"

        self.sessions.append({
            "session_id": session_id,
            "start_time": datetime.now(),
            "ip": ip,
            "browser": browser,
            "os": os_name,
            "device": device
        })


    def register_request(self, session_id: str, path: str, method: str):
        """Store one HTTP request (page visited) of a session"""

        self.requests.append({
            "session_id": session_id,
            "timestamp": datetime.now(),
            "path": path,
            "method": method
        })
    

    def save_query_terms(self, terms: str, session_id: str) -> int:
        """ store a new seacrh query and return its unique search_id.
        saves the query text, num of terms and its order inside the session"""

        search_id = next(self._query_counter)
        order = sum(1 for q in self.queries if q["session_id"] == session_id) + 1

        self.queries.append({
            "search_id": search_id,
            "session_id": session_id,
            "timestamp": datetime.now(),
            "query": terms,
            "n_terms": len(terms.split()),
            "n_results": None,  # to be updated later
            "order_in_session": order
        })

        return search_id

    def update_query_results(self, search_id: int, n_results: int):
        """store how many results a query returned (0 = that the engine found nothing)"""
        for q in self.queries:
            if q["search_id"] == search_id:
                q["n_results"] = n_results
                break
    
    def register_click(self, session_id: str, search_id, doc_id: str, rank) -> int:
        """
        Store a click on a document. search_id links it to the query it came from and rank
        is the position of the document in the results (None if the page was opened directly,
        not from a results list). Returns a unique click_id for the click.
        """

        click_id = next(self._click_counter)

        self.clicks.append({
            "click_id": click_id,
            "session_id": session_id,
            "search_id": search_id,
            "doc_id": doc_id,
            "rank": rank,
            "timestamp": datetime.now(),
            "dwell_time": None  # to be updated later
        })

        return click_id

    def update_dwell_time(self, click_id: int, seconds: float):
        """store how many seconds the user stayed on the doc page"""
        for c in self.clicks:
            if c["click_id"] == click_id:
                c["dwell_time"] = seconds
                break
    
    
    def plot_number_of_views(self):
        # Prepare data
        data = [{'Document ID': doc_id, 'Number of Views': count} for doc_id, count in self.fact_clicks.items()]
        df = pd.DataFrame(data)
        # Create Altair chart
        chart = alt.Chart(df).mark_bar().encode(
            x='Document ID',
            y='Number of Views'
        ).properties(
            title='Number of Views per Document'
        )
        # Render the chart to HTML
        return chart.to_html()

    # DASHBOARD --> KPIs and charts

    def get_kpis(self) -> list:
        #compute the kpis shown at the top of the dashboard

        queries = pd.DataFrame(self.queries)
        clicks = pd.DataFrame(self.clicks)
        n_sessions = len(self.sessions)
        n_queries = len(queries)
        n_clicks = len(clicks)

        #clicks that come from a results page (they have a search_id and a rank) 
        result_clicks = clicks.dropna(subset=["search_id", "rank"]) if n_clicks else clicks

        #click-through rate: % of searches with at least one click on a result
        if n_queries:
            clicked_searches = set(result_clicks["search_id"]) if len(result_clicks) else set()
            ctr = len(clicked_searches & set(queries["search_id"])) / n_queries * 100
            ctr_txt = f"{ctr:.0f}%"
            zero_txt = f"{(queries['n_results'] == 0).mean() * 100:.0f}%"
            terms_txt = f"{queries['n_terms'].mean():.1f}"
        else:
            ctr_txt = zero_txt = terms_txt = "-"

        avg_rank_txt = f"{result_clicks['rank'].mean():.1f}" if len(result_clicks) else "-"
        dwell = clicks["dwell_time"].dropna() if n_clicks else pd.Series(dtype=float)
        dwell_txt = f"{dwell.mean():.0f} s" if len(dwell) else "-"

        return [
            {"label": "Sessions", "value": n_sessions,
             "help": "Different visitors (browser cookies)."},
            {"label": "Page requests", "value": len(self.requests),
             "help": "Pages loaded in total."},
            {"label": "Searches", "value": n_queries,
             "help": "Queries submitted to the engine."},
            {"label": "Searches per session",
             "value": f"{n_queries / n_sessions:.1f}" if n_sessions else "-",
             "help": "How many queries a visitor makes on average."},
            {"label": "Avg. terms per query", "value": terms_txt,
             "help": "Longer queries = more specific information needs."},
            {"label": "Zero-result searches", "value": zero_txt,
             "help": "Queries where the engine found nothing."},
            {"label": "Clicks on results", "value": n_clicks,
             "help": "Product pages opened."},
            {"label": "Click-through rate", "value": ctr_txt,
             "help": "% of searches with at least one click."},
            {"label": "Avg. rank of clicked results", "value": avg_rank_txt,
             "help": "Close to 1 = users find what they want at the top."},
            {"label": "Avg. dwell time", "value": dwell_txt,
             "help": "Time spent on a product page after clicking it."},
        ]

    @staticmethod
    def _bar(df: pd.DataFrame, x: str, y: str, x_title: str, y_title: str,
             horizontal: bool = False, x_type: str = "N", sort=None) -> str:
        """Build a simple Altair bar chart and return it as a Vega-Lite JSON spec."""
        if horizontal:
            enc = dict(
                y=alt.Y(f"{x}:{x_type}", title=x_title, sort=sort if sort else "-x"),
                x=alt.X(f"{y}:Q", title=y_title),
            )
        else:
            enc = dict(
                x=alt.X(f"{x}:{x_type}", title=x_title, sort=sort),
                y=alt.Y(f"{y}:Q", title=y_title),
            )
        chart = (alt.Chart(df).mark_bar()
                 .encode(**enc, tooltip=[x, y])
                 .properties(width="container", height=250))
        return chart.to_json()

    def _count_chart(self, values: pd.Series, name: str, title: str, horizontal=False, x_type="N", sort=None, top=None) -> str:
        """count the values of a column and draw them as a bar chart."""
        counts = values.value_counts()
        if top:
            counts = counts.head(top)
        df = counts.rename_axis(name).reset_index(name="count")
        return self._bar(df, name, "count", title, "Count", horizontal, x_type, sort)

    def get_charts(self, corpus: dict) -> list:
        """build all the dashboard charts"""
        sessions = pd.DataFrame(self.sessions)
        requests = pd.DataFrame(self.requests)
        queries = pd.DataFrame(self.queries)
        clicks = pd.DataFrame(self.clicks)
        result_clicks = clicks.dropna(subset=["search_id", "rank"]) if len(clicks) else clicks

        charts = []

        def add(chart_id, title, explanation, spec_function, has_data):
            charts.append({
                "id": chart_id,
                "title": title,
                "explanation": explanation,
                "spec": spec_function() if has_data else None,
            })

        # 1. Clicks by rank position
        add("clicks_by_rank", "Clicks by position in the results",
            "How many clicks each position of the results list receives. If most clicks are "
            "in the first positions, users trust the ranking (or do not scroll: position bias).",
            lambda: self._count_chart(result_clicks["rank"].astype(int), "rank",
                                      "Position in results", x_type="O"),
            len(result_clicks) > 0)

        # 2. Most frequent queries
        add("top_queries", "Top 10 queries",
            "The most frequent searches (lowercased). They show what users look for most in the shop.",
            lambda: self._count_chart(queries["query"].str.lower().str.strip(), "query",
                                      "Query", horizontal=True, top=10),
            len(queries) > 0)

        # 3. Most clicked products
        def top_docs():
            titles = clicks["doc_id"].map(
                lambda pid: corpus[pid].title[:45] if pid in corpus else pid)
            return self._count_chart(titles, "product", "Product", horizontal=True, top=10)
        add("top_docs", "Top 10 clicked products",
            "Products opened most often from the results. Popular products could be promoted.",
            top_docs, len(clicks) > 0)

        # 4. Number of terms per query
        add("terms_per_query", "Number of terms per query",
            "Distribution of query length. Short queries (1-2 terms) are vague; longer ones "
            "describe a precise need (and are more likely to return 0 results with AND retrieval).",
            lambda: self._count_chart(queries["n_terms"], "terms", "Number of terms",
                                      x_type="O"),
            len(queries) > 0)

        # 5. Dwell time
        def dwell_chart():
            dwell = clicks["dwell_time"].dropna()
            groups = pd.cut(dwell, bins=[0, 5, 30, 120, float("inf")],
                            labels=["< 5 s", "5-30 s", "30 s - 2 min", "> 2 min"],
                            include_lowest=True).astype(str)
            order = ["< 5 s", "5-30 s", "30 s - 2 min", "> 2 min"]
            return self._count_chart(groups, "dwell", "Time on the product page", sort=order)
        add("dwell_time", "Dwell time on product pages",
            "Time between opening a product and leaving its page. Very short visits (< 5 s) "
            "suggest the result was not relevant; long visits suggest real interest.",
            dwell_chart, len(clicks) > 0 and clicks["dwell_time"].notna().any())

        # 6. Activity by hour of the day
        add("activity_by_hour", "Activity by hour of the day",
            "Number of page requests per hour. Shows when the website is used the most.",
            lambda: self._count_chart(pd.to_datetime(requests["timestamp"]).dt.hour,
                                      "hour", "Hour of the day", x_type="O"),
            len(requests) > 0)

        # 7. Pages visited
        add("pages", "Pages visited",
            "Requests per page of the website: how users move between search, results and products.",
            lambda: self._count_chart(requests["path"], "page", "Page", horizontal=True),
            len(requests) > 0)

        # 8-10. User context
        for col, title, explanation in [
            ("browser", "Browsers", "Browser used by each session (from the User-Agent)."),
            ("os", "Operating systems", "Operating system of each session (from the User-Agent)."),
            ("device", "Device type", "Computer, mobile or tablet, detected from the User-Agent."),
        ]:
            add(col, title, explanation,
                lambda col=col, title=title: self._count_chart(sessions[col], col, title),
                len(sessions) > 0)

        return charts

    def get_zero_result_queries(self) -> list:
        """Queries that returned no results (useful to detect what the engine misses)."""
        return [q for q in self.queries if q["n_results"] == 0]    # ------------------------------------------------------------------
    # SESSIONS (bonus): physical sessions (sit-downs) and missions
    # ------------------------------------------------------------------

    # a new physical session starts after 30 minutes without any request
    SESSION_TIMEOUT = pd.Timedelta(minutes=30)

    def get_queries_with_missions(self) -> pd.DataFrame:
        """
        Return the queries table with two extra columns:
        - physical_session: sit-down the query belongs to. A cookie session is split
          into a new physical session after 30 min without any request.
        - mission: logical session inside a sit-down. A query starts a new mission
          when it shares no (stemmed) term with the queries of the current mission.
        """
        if not self.queries or not self.requests:
            return pd.DataFrame()

        # 1. physical sessions, computed from ALL requests (searching, reading products...)
        req = pd.DataFrame(self.requests).sort_values(["session_id", "timestamp"])
        gap = req.groupby("session_id")["timestamp"].diff()
        new_sitdown = gap.isna() | (gap > self.SESSION_TIMEOUT)
        req["sitdown_number"] = new_sitdown.astype(int).groupby(req["session_id"]).cumsum()
        req["physical_session"] = req["session_id"].str[:8] + "-" + req["sitdown_number"].astype(str)

        # assign each query to the physical session of the last request before it
        q = pd.DataFrame(self.queries).sort_values("timestamp")
        q = pd.merge_asof(q, req.sort_values("timestamp")[["timestamp", "session_id", "physical_session"]],
                          on="timestamp", by="session_id", direction="backward")

        # 2. missions inside each physical session: consecutive queries sharing terms
        q = q.sort_values(["physical_session", "timestamp"])
        missions = []
        for _, group in q.groupby("physical_session", sort=False):
            mission_number, mission_terms = 0, set()
            for query in group["query"]:
                terms = set(build_terms(query))
                if not mission_terms or not (terms & mission_terms):
                    mission_number += 1          # new goal -> new mission
                    mission_terms = set()
                mission_terms |= terms
                missions.append(mission_number)
        q["mission"] = q["physical_session"] + "-M" + pd.Series(missions, index=q.index).astype(str)
        return q

    def get_session_kpis(self) -> list:
        """Key indicators about physical sessions and missions."""
        q = self.get_queries_with_missions()
        if q.empty:
            return []
        n_sitdowns = q["physical_session"].nunique()
        n_missions = q["mission"].nunique()
        return [
            {"label": "Physical sessions (sit-downs)", "value": n_sitdowns,
             "help": "A new one starts after 30 min without activity."},
            {"label": "Missions", "value": n_missions,
             "help": "Groups of consecutive queries with the same goal."},
            {"label": "Missions per sit-down", "value": f"{n_missions / n_sitdowns:.1f}",
             "help": "How many different goals a user has in one visit."},
            {"label": "Queries per mission", "value": f"{len(q) / n_missions:.1f}",
             "help": "More than 1 = users reformulate to reach their goal."},
        ]

    def get_missions_table(self) -> list:
        """One row per mission: its sit-down, the sequence of queries and clicks obtained."""
        q = self.get_queries_with_missions()
        if q.empty:
            return []
        clicks = pd.DataFrame(self.clicks)
        clicks_per_search = clicks["search_id"].value_counts() if len(clicks) else pd.Series(dtype=int)

        rows = []
        for mission, group in q.groupby("mission", sort=False):
            rows.append({
                "physical_session": group["physical_session"].iloc[0],
                "mission": mission.split("-")[-1],
                "start": group["timestamp"].min(),
                "queries": " → ".join(group["query"]),
                "n_queries": len(group),
                "n_clicks": int(sum(clicks_per_search.get(s, 0) for s in group["search_id"])),
            })
        return rows

    def get_queries_per_mission_chart(self):
        """Bar chart: how many missions have 1, 2, 3... queries (None if no data)."""
        q = self.get_queries_with_missions()
        if q.empty:
            return None
        return self._count_chart(q.groupby("mission").size(), "queries_in_mission",
                                 "Queries in the mission", x_type="O")


class ClickedDoc:
    def __init__(self, doc_id, description, counter):
        self.doc_id = doc_id
        self.description = description
        self.counter = counter

    def to_json(self):
        return self.__dict__

    def __str__(self):
        """
        Print the object content as a JSON string
        """
        return json.dumps(self)
