import os
from json import JSONEncoder

import httpagentparser  # for getting the user agent as json
from flask import Flask, render_template, session, redirect, url_for
from flask import request

from myapp.analytics.analytics_data import AnalyticsData, ClickedDoc
from myapp.search.load_corpus import load_corpus
from myapp.search.objects import Document, StatsDocument
from myapp.search.search_engine import SearchEngine
from myapp.generation.rag import RAGGenerator
from dotenv import load_dotenv
from myapp.search.algorithms import build_search_index

import uuid

load_dotenv()  # take environment variables from .env


# *** for using method to_json in objects ***
def _default(self, obj):
    return getattr(obj.__class__, "to_json", _default.default)(obj)
_default.default = JSONEncoder().default
JSONEncoder.default = _default
# end lines ***for using method to_json in objects ***


# instantiate the Flask application
app = Flask(__name__)

# random 'secret_key' is used for persisting data in secure cookie
app.secret_key = os.getenv("SECRET_KEY")
# open browser dev tool to see the cookies
app.session_cookie_name = os.getenv("SESSION_COOKIE_NAME")
# instantiate our search engine
search_engine = SearchEngine()
# instantiate our in memory persistence
analytics_data = AnalyticsData()
# instantiate RAG generator
rag_generator = RAGGenerator()

@app.before_request
def before_request():
    """ runs before every request: assigns a session id to new visitors (stored in the Flask session cookie),
    saves their user context once, and logs the request."""

    #do not tracj static files or browser's favicon request
    if request.path.startswith('/static') or request.path in ('/favicon.ico', '/dwell'):
        return

    # keep the cookie for 31 days (not only until the browser closes), so the same
    # visitor can be followed across several sit-downs
    session.permanent = True

    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())

    session_id = session["session_id"]

    #register the session if we don't have it yet
    if not analytics_data.session_exists(session_id):
        analytics_data.register_session(session_id, request.remote_addr, request.headers.get('User-Agent'))

    analytics_data.register_request(session_id, request.path, request.method)

# load documents corpus into memory.
full_path = os.path.realpath(__file__)
path, filename = os.path.split(full_path)
file_path = path + "/" + os.getenv("DATA_FILE_PATH")
corpus = load_corpus(file_path)
# Log first element of corpus to verify it loaded correctly:
print("\nCorpus is loaded... \n First element:\n", list(corpus.values())[0])
# build the search index ONCE at startup, so the first search is not slow
print("Building search index...")
search_engine.index_data = build_search_index(corpus)
print("Search index ready.")


# Home URL "/"
@app.route('/')
def index():
    print("starting home url /...")

    # flask server creates a session by persisting a cookie in the user's browser.
    # the 'session' object keeps data between multiple requests. Example:
    session['some_var'] = "Some value that is kept in session"

    user_agent = request.headers.get('User-Agent')
    print("Raw user browser:", user_agent)

    user_ip = request.remote_addr
    agent = httpagentparser.detect(user_agent)

    print("Remote IP: {} - JSON user browser {}".format(user_ip, agent))
    print(session)
    return render_template('index.html', page_title="Welcome")


@app.route('/search', methods=['GET'])
def search_form_post():
    
    search_query = request.args.get('search-query', '').strip()

    # empty search -> back to the home page
    if not search_query:
        return redirect(url_for('index'))

    session['last_search_query'] = search_query

    search_id = analytics_data.save_query_terms(search_query, session['session_id'])

    results = search_engine.search(search_query, search_id, corpus)

    analytics_data.update_query_results(search_id, len(results))

    # generate RAG response based on user query and retrieved results
    rag_response = rag_generator.generate_response(search_query, results)
    print("RAG response:", rag_response)

    found_count = len(results)
    session['last_found_count'] = found_count

    print(session)

    return render_template('results.html', results_list=results, page_title="Results", found_counter=found_count, rag_response=rag_response, search_id=search_id)


@app.route('/doc_details', methods=['GET'])
def doc_details():
    """
    Show the full information of a product (document details page).
    Only counts a click if the product exists in the corpus.
    """
    # get the product id from the URL (?pid=...)
    clicked_doc_id = request.args.get("pid")
    print("click in id={}".format(clicked_doc_id))

    # get the search and the position the click came from (?search_id=...&rank=...)
    search_id = request.args.get("search_id", type=int)
    rank = request.args.get("rank", type=int)
    print("click from search_id={} at rank={}".format(search_id, rank))

    # look for the product in the corpus BEFORE counting the click
    doc = corpus.get(clicked_doc_id)

    # if the product does not exist, show the "not found" page without counting the click
    if doc is None:
        return render_template('doc_details.html', doc=None)

    analytics_data.fact_clicks[clicked_doc_id] = analytics_data.fact_clicks.get(clicked_doc_id, 0) + 1

    #register the click in the clicks table
    click_id = analytics_data.register_click(session['session_id'], search_id, clicked_doc_id, rank)
    print("click {} on doc={} from search_id={} at rank={}".format(click_id, clicked_doc_id, search_id, rank))

    return render_template('doc_details.html', doc=doc, click_id=click_id)


@app.route('/stats', methods=['GET'])
def stats():
    """
    Show simple statistics example. ### Replace with your dashboard ###
    :return:
    """

    docs = []
    for doc_id in analytics_data.fact_clicks:
        row: Document = corpus[doc_id]
        count = analytics_data.fact_clicks[doc_id]
        doc = StatsDocument(pid=row.pid, title=row.title, description=row.description, url=row.url, count=count)
        docs.append(doc)
    
    # simulate sort by ranking
    docs.sort(key=lambda doc: doc.count, reverse=True)
    return render_template('stats.html', clicks_data=docs)


@app.route('/dashboard', methods=['GET'])
def dashboard():
    return render_template('dashboard.html', page_title="Dashboard",
                           kpis=analytics_data.get_kpis(),
                           charts=analytics_data.get_charts(corpus),
                           zero_queries=analytics_data.get_zero_result_queries(),
                           session_kpis=analytics_data.get_session_kpis(),
                           missions=analytics_data.get_missions_table(),
                           missions_chart=analytics_data.get_queries_per_mission_chart())



@app.route('/dwell', methods=['POST'])
def dwell():
    """
    Receives the dwell time from the document page (sent by JavaScript when the
    user leaves the page) and stores it in the corresponding click.
    """
    click_id = request.args.get("click_id", type=int)
    seconds = request.args.get("seconds", type=float)
    if click_id is not None and seconds is not None:
        analytics_data.update_dwell_time(click_id, round(seconds, 1))
    return "", 204   # 204 = OK, no content to return

#ELIMINA
@app.route('/analytics_raw', methods=['GET'])
def analytics_raw():
    return {
        "sessions": analytics_data.sessions,
        "last_requests": analytics_data.requests[-20:],
        "queries": analytics_data.queries,
        "clicks": analytics_data.clicks,
    }


if __name__ == "__main__":
    app.run(port=8088, host="0.0.0.0", threaded=True, debug=os.getenv("DEBUG"))
