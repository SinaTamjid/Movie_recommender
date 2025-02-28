from flask import Flask,request,jsonify
from py.NLP.Movie_recommender.Movie_recommender.rec_model import recommendation as rec

app=Flask(__name__)

@app.route('/recommend',methods=['POST'])
def recommend():
               movie_title=request.json['movie_title']
               recommendation=rec(movie_title)
               return jsonify(recommendation.tolist())

if __name__=='__main__':
               app.run(debug=True)