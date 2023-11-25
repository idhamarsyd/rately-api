from flask import Flask, jsonify, request
import enum
from typing import List
import json
import pandas as pd
from datetime import datetime, timedelta, timezone
from sqlalchemy import Integer, String, Text, MetaData, Enum, select, ForeignKey, create_engine, insert
from sqlalchemy.orm import sessionmaker, DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.exc import IntegrityError
from flask_jwt_extended import create_access_token, JWTManager, jwt_required, get_jwt_identity, unset_jwt_cookies
from werkzeug.security import generate_password_hash, check_password_hash

from preprocessing import Preprocessing
from information_gain import InformationGain

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

engine = create_engine('mysql+pymysql://root:@localhost/movie-review', echo=True)
Session = sessionmaker(bind=engine)
session = Session()

app = Flask(__name__)

app.config["JWT_SECRET_KEY"] = "please-remember-to-change-me"
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(hours=4)
app.config['JWT_TOKEN_LOCATION'] = ['headers', 'query_string']
app.config['JWT_BLACKLIST_ENABLED'] = True
jwt = JWTManager(app)

class Base(DeclarativeBase):
    pass

class Sentiments(enum.Enum):
    POSITIVE = 1
    NEGATIVE = 2

class Movies(Base):
    __tablename__ = "movies"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(255))
    year: Mapped[int]
    genre: Mapped[str] = mapped_column(String(255))
    description: Mapped[str] = mapped_column(Text)
    cover: Mapped[str] = mapped_column(String(255))
    trailer: Mapped[str] = mapped_column(String(255))

    comments: Mapped[List["Comments"]] = relationship(
         back_populates="movie_title", cascade="all, delete-orphan"
     )

    @classmethod
    def get_all_movies(cls, session: Session) -> list:
        movies = session.scalars(select(Movies)).all()
        return movies
    
    @classmethod
    def search_movie(cls, session: Session, param: str) -> list:
        movies = session.scalars(select(Movies).filter(Movies.title.contains(param))).all()
        return movies
    
    @classmethod
    def get_movies_by_id(cls, session: Session, id: int) -> list:
        movie = session.scalars(select(Movies).filter_by(id=id)).first()
        if movie:
            return movie.to_dict()
        else:
            return {"error": "Movie not found."}
        # return movie
    
    @classmethod
    def delete_movie(cls, session: Session, id: int) -> list:
        try:
            movie = session.scalars(select(Movies).filter_by(id=id)).first()
            if movie:
                session.delete(movie)
                session.commit()
                return {"status": "success", "msg": "Film berhasil dihapus."}
            else:
                return {"status": "error", "msg": "Data film tidak ditemukan."}

        except Exception as e:
            # Log the exception for debugging purposes
            print(f"Error deleting movie: {e}")
            session.rollback()
            return {"status": "error", "msg": "An error occurred while deleting the movie."}
        
    @classmethod
    def add_movie(cls, session: Session, data: dict) -> dict:
        try:
            check_movie = session.scalars(select(Movies).filter_by(title=data.get("title"))).first()

            if check_movie:
                return {"status": "error", "msg": "Movie with the same title already exists."}
            
            new_movie = cls(**data)
            session.add(new_movie)
            session.commit()
            return {"status": "success", "msg": "Movie successfully added."}
        except IntegrityError as e:
            print(f"IntegrityError adding movie: {e}")
            session.rollback()
            return {"status": "error", "msg": "An error occurred while adding the movie. Integrity error."}

        except Exception as e:
            print(f"Error adding movie: {e}")
            session.rollback()
            return {"status": "error", "msg": "An error occurred while adding the movie."}
    
    @classmethod
    def update_movie(cls, session: Session, id: int, data: dict) -> dict:
        try:
            movie = session.scalars(select(Movies).filter_by(id=id)).first()

            if not movie:
                return {"status": "error", "msg": "Movie not found."}
            
            for key, value in data.items():
                setattr(movie, key, value)

            session.commit()
            return {"status": "success", "msg": "Movie successfully updated."}
        
        except Exception as e:
            print(f"Error updating movie: {e}")
            session.rollback()
            return {"status": "error", "msg": "An error occurred while updating the movie."}
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "year": self.year,
            "genre": self.genre,
            "description": self.description,
            "cover": self.cover,
            "trailer": self.trailer
        }
    
    def __repr__(self) -> str:
        return str(self.to_dict())
    
class Comments(Base):
    __tablename__ = "comments"

    id: Mapped[int] = mapped_column(primary_key=True)
    comment: Mapped[str] = mapped_column(Text)
    movie: Mapped[int] = mapped_column(ForeignKey("movies.id", onupdate="CASCADE", ondelete="CASCADE"))
    label: Mapped[Sentiments] = mapped_column(Enum(Sentiments))

    movie_title: Mapped["Movies"] = relationship(back_populates="comments")

    def sentiment_enum_serializer(obj):
        if isinstance(obj, Sentiments):
          return obj.name
        raise TypeError("Object not serializable")

    @classmethod
    def get_all_comments(cls, session: Session) -> list:
        stmt = (
            select(Comments)
            .join(Comments.movie_title)
            .where(Movies.id == Comments.movie)
        )
        comments = session.scalars(stmt).all()
        return comments
    
    @classmethod
    def delete_comment(cls, session: Session, id: int) -> list:
        try:
            comment = session.scalars(select(Comments).filter_by(id=id)).first()
            if comment:
                session.delete(comment)
                session.commit()
                return {"status": "success", "msg": "Komentar berhasil dihapus."}
            else:
                return {"status": "error", "msg": "Data komentar tidak ditemukan."}

        except Exception as e:
            # Log the exception for debugging purposes
            print(f"Error deleting comment: {e}")
            session.rollback()
            return {"status": "error", "msg": "An error occurred while deleting the comment."}
        
    @classmethod
    def add_comment(cls, session: Session, data: dict) -> dict:
        try:
            check_movie = session.scalars(select(Movies).filter_by(title=data.get("movie"))).first()

            if not check_movie:
                return {"status": "error", "msg": "Data film tidak ditemukan"}
            
            # new_movie = cls(**data)
            new_comment = Comments(
                comment=data['comment'],
                movie=check_movie.id,
                label=data['label'],
            )
            session.add(new_comment)
            session.commit()
            return {"status": "success", "msg": "Comment successfully added."}
        except IntegrityError as e:
            print(f"IntegrityError adding movie: {e}")
            session.rollback()
            return {"status": "error", "msg": "An error occurred while adding the comment. Integrity error."}

        except Exception as e:
            print(f"Error adding movie: {e}")
            session.rollback()
            return {"status": "error", "msg": "An error occurred while adding the comment."}
    
    @classmethod
    def update_comment(cls, session: Session, id: int, data: dict) -> dict:
        try:
            comment = session.scalars(select(Comments).filter_by(id=id)).first()

            if not comment:
                return {"status": "error", "msg": "Comment not found"}

            check_movie = session.scalars(select(Movies).filter_by(title=data['movie'])).first()

            if not check_movie:
                return {"status": "error", "msg": "Data film tidak ditemukan"}
            
            comment.comment = data['comment']
            comment.movie = check_movie.id
            comment.label = data['label']

            session.commit()
            return {"status": "success", "msg": "Comment successfully updated."}
        
        except Exception as e:
            print(f"Error updating comment: {e}")
            session.rollback()
            return {"status": "error", "msg": "An error occurred while updating the comment."}

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "comment": self.comment,
            "movie": self.movie,
            "label": self.label,
        }
    
    def __repr__(self) -> str:
        return str(self.to_dict())\
        
class Classification(Base):
    __tablename__ = "classifications"

    id: Mapped[int] = mapped_column(primary_key=True)
    comment_id: Mapped[int] = mapped_column(ForeignKey("comments.id", onupdate="CASCADE", ondelete="CASCADE"))
    comment: Mapped[str] = mapped_column(Text)
    movie: Mapped[int] = mapped_column(ForeignKey("movies.id", onupdate="CASCADE", ondelete="CASCADE"))
    classification: Mapped[Sentiments] = mapped_column(Enum(Sentiments))

    movie_title: Mapped["Movies"] = relationship(back_populates="comments")

    def sentiment_enum_serializer(obj):
        if isinstance(obj, Sentiments):
          return obj.name
        raise TypeError("Object not serializable")
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "comment": self.comment,
            "comment_id": self.comment_id,
            "movie": self.movie,
            "classification": self.classification,
        }
    
    def __repr__(self) -> str:
        return str(self.to_dict())\
    
    @classmethod
    def get_all_comments_classiication(cls, session: Session) -> list:
        stmt = (
            select(Classification)
            .join(Classification.movie_title)
            .where(Movies.id == Classification.movie)
        )
        comments = session.scalars(stmt).all()
        return comments

    @classmethod
    def save_classifications(cls, session: Session, data: dict) -> dict:
        try:
            session.execute(insert(Classification), data)
            session.commit()
            return {"status": "success", "msg": "Comments successfully added."}
        except IntegrityError as e:
            print(f"IntegrityError adding movie: {e}")
            session.rollback()
            return {"status": "error", "msg": "An error occurred while adding the comment. Integrity error."}

        except Exception as e:
            print(f"Error adding movie: {e}")
            session.rollback()
            return {"status": "error", "msg": "An error occurred while adding the comment."}


    
class Users(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    password: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)

    @classmethod
    def validate_user(cls, session: Session, data: dict) -> list:
        try:
            user = session.scalars(select(Users).filter_by(email=data['email'])).first()

            if not user or not check_password_hash(user.password, data['password']):
                return {"status": "error", "msg": "Invalid email or password"}
            
            return {"status": "success", "msg": "Login success"}
        
        except Exception as e:
            print(f"Error login: {e}")
            session.rollback()
            return {"status": "error", "msg": "An error occurred while attempting to login."}

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "email": self.email,
            "password": self.password,
        }
    
    def __repr__(self) -> str:
        return str(self.to_dict())

Base.metadata.create_all(engine)
# Base.metadata.drop_all(engine)


new_user = Users(
    email="admin@hi.com",
    password=generate_password_hash("helloworld")
)

new_comment = Comments(
    comment="Great",
    movie=2,
    label="POSITIVE",
)

# session.add(new_comment)
# session.commit()
# session.add(new_user)
# session.commit()




@app.route('/token', methods=["POST"])
def create_token():
    data = request.get_json()
    validate = Users.validate_user(session=session, data=data)
    if validate['status'] == "error":
        return {"msg": "Wrong email or password"}, 401

    access_token = create_access_token(identity=data['email'])
    response = {"access_token":access_token}
    return response

@app.route("/logout", methods=["POST"])
def logout():
    response = jsonify({"msg": "logout successful"})
    unset_jwt_cookies(response)
    return response

@app.route("/movies")
def get_all_movies():
    movies_list = Movies.get_all_movies(session)
    movies_data = []
    for movie in movies_list:
        data = {
            "id": movie.id,
            "title": movie.title,
            "year": movie.year,
            "genre": movie.genre,
            "description": movie.description,
            "cover": movie.cover,
            "trailer": movie.trailer
        }
        movies_data.append(data)
    return jsonify(movies_data)

@app.route("/movie/<int:id>", methods=["GET"])
def get_movie_by_id(id):
    result = Movies.get_movies_by_id(session=session, id=id)
    if "error" in result:
        return jsonify({"error": result["error"]}), 404
    else:
        return jsonify(result), 200
    
@app.route("/search/<string:param>", methods=["GET"])
def search_movie(param):
    result = Movies.search_movie(session=session, param=param)
    movies_data = []
    for movie in result:
        data = {
            "id": movie.id,
            "title": movie.title,
            "year": movie.year,
            "genre": movie.genre,
            "description": movie.description,
            "cover": movie.cover,
            "trailer": movie.trailer
        }
        movies_data.append(data)
    return jsonify(movies_data)

@app.route("/delete_movie/<int:id>", methods=["DELETE"])
@jwt_required()
def delete_movie(id):
    result = Movies.delete_movie(session=session, id=id)
    if result["status"] == "success":
        return jsonify({"msg": result["msg"]}), 200
    else:
        return jsonify({"msg": result["msg"]}), 404

@app.route("/add_movie", methods=["POST"])
@jwt_required()
def add_movie():
    data = request.get_json()
    result = Movies.add_movie(session=session, data=data)
    if result["status"] == "success":
        return jsonify({"msg": result["msg"]}), 200
    else:
        return jsonify({"msg": result["msg"]}), 404

@app.route("/update_movie/<int:id>", methods=["PUT"])
@jwt_required()
def update_movie(id):
    data = request.get_json()
    result = Movies.update_movie(session=session, data=data, id=id)
    if result["status"] == "success":
        return jsonify({"msg": result["msg"]}), 200
    else:
        return jsonify({"msg": result["msg"]}), 404
    
@app.route("/comments")
def get_all_comments():
    comments_list = Comments.get_all_comments(session)
    comments_data = []
    for comment in comments_list:
        data = {
            "id": comment.id,
            "comment": comment.comment,
            "movie": comment.movie_title.title,
            "label": comment.label,
        }
        comments_data.append(data)
    data_json = json.dumps(comments_data, default=Comments.sentiment_enum_serializer)
    return data_json

@app.route("/delete_comment/<int:id>", methods=["DELETE"])
@jwt_required()
def delete_comment(id):
    result = Comments.delete_comment(session=session, id=id)
    if result["status"] == "success":
        return jsonify({"msg": result["msg"]}), 200
    else:
        return jsonify({"msg": result["msg"]}), 404

@app.route("/add_comment", methods=["POST"])
@jwt_required()
def add_comment():
    data = request.get_json()
    result = Comments.add_comment(session=session, data=data)
    if result["status"] == "success":
        return jsonify({"msg": result["msg"]}), 200
    else:
        return jsonify({"msg": result["msg"]}), 404

@app.route("/update_comment/<int:id>", methods=["PUT"])
@jwt_required()
def update_comment(id):
    data = request.get_json()
    result = Comments.update_comment(session=session, data=data, id=id)
    if result["status"] == "success":
        return jsonify({"msg": result["msg"]}), 200
    else:
        return jsonify({"msg": result["msg"]}), 404

@app.route("/classification", methods=["GET"])
# @jwt_required()
def preprocessing():
    # Get all comments
    comments = Comments.get_all_comments(session)

    vectorizer = TfidfVectorizer()    

    # Start preprocessing
    preprocessor = Preprocessing()
    results = []
    for data in comments:
        result = {
            "id": data.id,
            "result": preprocessor.preprocessing_function(data.comment)
            }
        results.append(result)
    
    # Map labels to numeric value
    label_mapping = {Sentiments.POSITIVE: 1, Sentiments.NEGATIVE: 0}
    labels = [label_mapping[comment.label] for comment in comments]

    # Get results from preprocessing
    texts = [" ".join(result["result"]) for result in results]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.3, random_state=42)


    # Start Information Gain
    X_train_tfidf = vectorizer.fit_transform(X_train)
    information_gain = mutual_info_classif(X_train_tfidf, y_train)
    # information_gain = IG.mutual_info_classif(X_train_tfidf.toarray(), y_train)
    feature_info_gain = list(zip(vectorizer.get_feature_names_out(), information_gain))
    sorted_feature_info_gain = sorted(feature_info_gain, key=lambda x: x[1], reverse=True)

    # Select top 10 features with highest IG
    top_n = 10
    selected_features = [feature for feature, _ in sorted_feature_info_gain[:top_n]]

    # Start TF-IDF
    vectorizer_selected = TfidfVectorizer(vocabulary=selected_features)
    X_train_selected = vectorizer_selected.fit_transform(X_train)

    # Train SVM model
    svm_model = SVC(kernel='rbf')
    svm_model.fit(X_train_selected, y_train)

    # Vectorize the testing data using TF-IDF with only the selected features
    X_test_selected = vectorizer_selected.transform(X_test)

    # Train without IG
    # svm_model_pure = SVC(kernel='rbf')
    # svm_model_pure.fit(X_train_tfidf, y_train)
    # X_test_selected_pure = vectorizer.transform(X_test)
    # y_pred_no_info_gain = svm_model_pure.predict(X_test_selected_pure)
    # accuracy_no_info_gain = accuracy_score(y_test, y_pred_no_info_gain)

    # Make predictions on the testing data
    y_pred = svm_model.predict(X_test_selected)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f'Accuracy (IG): {accuracy}')
    # print(f'Accuracy (Without IG): {accuracy_no_info_gain}')
    print('Classification Report:\n', report)

    return jsonify({"Accuracy": accuracy, "Classification Report": report})

    dense_array = X_train_selected.toarray()
    tfidf_list = dense_array.tolist()
    print(f"Top {top_n} Features with High Information Gain:")
    for feature, ig in sorted_feature_info_gain[:top_n]:
        print(f"{feature}: {ig}")

    # return jsonify(tfidf_list)

if __name__ == '__main__':
    app.run(debug=True)