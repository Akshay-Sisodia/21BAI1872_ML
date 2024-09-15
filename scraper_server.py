from flask import Flask, jsonify, request
from scraper import NewsScraper  # Adjust import as necessary

app = Flask(__name__)
scraper = NewsScraper()

@app.route('/status', methods=['GET'])
def status():
    """
    Status endpoint to check if the service is running.
    """
    return jsonify({"status": "Service is up and running"}), 200

@app.route('/search', methods=['GET'])
def search():
    """
    Search endpoint to scrape news articles based on a search term.
    Expects a query parameter 'term' for the search term.
    """
    term = request.args.get('term', 'latest')
    if not term:
        return jsonify({"error": "Missing 'term' query parameter"}), 400

    results_json, runtime = scraper.run(term)
    return jsonify({"results": results_json, "runtime": runtime}), 200

if __name__ == '__main__':
    app.run(debug=True)
