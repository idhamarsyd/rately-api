from flask import Flask, jsonify, request
import enum
from typing import List
import json
import pandas as pd
from datetime import datetime, timedelta, timezone
from sqlalchemy import Integer, String, Text, MetaData, Enum, delete, func, select, ForeignKey, create_engine, insert, funcfilter, and_
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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from confusion import calculate_accuracy, calculate_precision_recall_f1


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
    NEUTRAL = 0
    NEGATIVE = -1

class DataType(enum.Enum):
    TRAINING = 0
    TESTING = 1

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
    classifications: Mapped[List["Classification"]] = relationship(
        back_populates="movie_title", cascade="all, delete-orphan"
    )

    @classmethod
    def get_all_movies(cls, session: Session) -> list:
        movies = session.scalars(select(Movies)).all()
        return movies
    
    @classmethod
    def get_classified_movies(cls, session: Session) -> list:
        query = (
            select(Movies)
            .join(Classification, Movies.id == Classification.movie)
            .distinct(Movies.id)
        )
        movies = session.scalars(query).all()
        return movies
    
    
    @classmethod
    def search_movie(cls, session: Session, param: str) -> list:
        query = (
            select(Movies)
            .filter(Movies.title.contains(param))
            .join(Classification, Movies.id == Classification.movie)
            .distinct(Movies.id)
        )
        movies = session.scalars(query).all()
        return movies
    
    @classmethod
    def get_movies_by_id(cls, session: Session, id: int) -> list:
        query = (
            select(Movies)
            .filter_by(id=id)
            .join(Classification, Movies.id == Classification.movie)
            .distinct(Movies.id)
        )
        movie = session.scalars(query).first()
        data = {
            "cover": movie.cover,
            "description": movie.description,
            "genre": movie.genre,
            "id": movie.id,
            "title": movie.title,
            "trailer": movie.trailer,
            "year": movie.year,
            "comments": Classification.get_comments(session=session, id=id)
        }
        # movie = session.scalars(select(Movies).filter_by(id=id)).first()
        if movie:
            return data
        else:
            return {"error": "Movie not found."}
    
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
    category: Mapped[DataType] = mapped_column(Enum(DataType))

    movie_title: Mapped["Movies"] = relationship(back_populates="comments")

    def sentiment_enum_serializer(obj):
        if isinstance(obj, enum.Enum):
          return obj.name
        raise TypeError("Object not serializable")

    @classmethod
    def get_all_comments(cls, session: Session) -> list:
        try:
            stmt = (
                select(Comments)
                .join(Comments.movie_title)
                .where(Movies.id == Comments.movie)
            )
            comments = session.scalars(stmt).all()
            return comments
        except Exception as e:
            print("Error fetching comments.")
            session.rollback()

    @classmethod
    def get_training(cls, session: Session) -> list:
        try:
            stmt = (
                select(Comments)
                .join(Comments.movie_title)
                .where(and_(Movies.id == Comments.movie, Comments.category == DataType.TRAINING))
            )
            comments = session.scalars(stmt).all()
            return comments
        except Exception as e:
            print(f"Error fetching training data: {e}")
            session.rollback()
        
    @classmethod
    def get_testing(cls, session: Session) -> list:
        try:
            stmt = (
                select(Comments)
                .join(Comments.movie_title)
                .where(and_(Movies.id == Comments.movie, Comments.category == DataType.TESTING))
            )
            comments = session.scalars(stmt).all()
            return comments
        except Exception as e:
            print(f"Error fetching testing data: {e}")
            session.rollback()
        
    @classmethod
    def validate_label(cls, comment_id: int, session: Session) -> str:
        try:
            comment = session.query(cls).filter_by(id=comment_id).first()
            if comment:
                classification = (
                    session.query(Classification)
                    .filter_by(comment_id=comment_id)
                    .first()
                )
                return "SESUAI" if comment.label == classification.classification else "TIDAK SESUAI"
            else:
                return "TIDAK SESUAI"  # Assuming not found should be treated as "TIDAK SESUAI"
        except Exception as e:
            print(f"Error in validating label for comment_id {comment_id}: {str(e)}")
            session.rollback()
            return "GAGAL VALIDASI"
    
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
                category=data['category']
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
            comment.category = data['category']

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

    movie_title: Mapped["Movies"] = relationship(back_populates="classifications")

    def sentiment_enum_serializer(obj):
        if isinstance(obj, Sentiments):
            return obj.name
        elif isinstance(obj, Classification):
            return obj.to_dict()  # Use the to_dict method you defined
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
    def get_all_classifications(cls, session: Session) -> list:
        try:
            stmt = (
                select(Classification)
                .join(Classification.movie_title)
                .where(Movies.id == Classification.movie)
            )
            comments = session.scalars(stmt).all()
            return comments
        except Exception as e:
            print("Erorr in fetching classifications.")
            session.rollback()
            return {"status": "error", "msg": "An error occurred while fetching the classification."}
        
    @classmethod
    def get_comments(cls, session: Session, id: int) -> list:
        try:
            stmt = (
                select(Classification)
                .join(Classification.movie_title)
                .where(id == Classification.movie)
            )
            comments = session.scalars(stmt).all()
            return comments
        except Exception as e:
            print("Erorr in fetching classifications.")
            session.rollback()
            return {"status": "error", "msg": "An error occurred while fetching the classification."}

    @classmethod
    def save_classifications(cls, session: Session, data: dict) -> dict:
        try:
            cls.clear_classification(session)
            session.execute(insert(Classification), data)
            session.commit()
            return {"status": "success", "msg": "Classification successfully added."}
        except IntegrityError as e:
            print(f"IntegrityError adding classification: {e}")
            session.rollback()
            return {"status": "error", "msg": "An error occurred while adding the classiication. Integrity error."}

        except Exception as e:
            print(f"Error adding classification: {e}")
            session.rollback()
            return {"status": "error", "msg": "An error occurred while adding the classification."}

    @classmethod
    def clear_classification(cls, session: Session):
        try:
            session.query(Classification).delete()
            session.commit()
        except Exception as e:
            print(f"Error deleting classification: {e}")
            session.rollback()
            return {"status": "error", "msg": "An error occurred while deleting the classification."}

class Metrics(Base):
    __tablename__ = "metrics"

    id: Mapped[int] = mapped_column(primary_key=True)
    accuracy: Mapped[float] = mapped_column()
    precision: Mapped[float] = mapped_column()
    recall: Mapped[float] = mapped_column()
    f1_score: Mapped[float] = mapped_column()
    
    @classmethod
    def get_metrics(cls, session: Session) -> list:
        try:
            metrics = session.scalars(select(Metrics)).first()
            return metrics
        except Exception as e:
            print("Error getting metrics.")
            session.rollback()
            
    
    @classmethod
    def update_metrics(cls, session: Session, data: dict) -> dict:
        try:
            metrics = session.scalars(select(Metrics)).first()

            if not metrics:
                session.execute(insert(Metrics), data)
                session.commit()
                return {"status": "success", "msg": "Metrics updated!"}

            metrics.accuracy = data['accuracy']
            metrics.precision = data['precision']
            metrics.recall = data['recall']
            metrics.f1_score = data['f1_score']

            session.commit()
            
            return {"status": "success", "msg": "Metrics updated!"}
        except IntegrityError as e:
            print(f"IntegrityError adding classification: {e}")
            session.rollback()
            return {"status": "error", "msg": "An error occurred while updating the metrics. Integrity error."}

        except Exception as e:
            print(f"Error adding classification: {e}")
            session.rollback()
            return {"status": "error", "msg": "An error occurred while updating the metrics."}

    @classmethod
    def clear_metrics(cls, session: Session):
        try:
            session.query(Metrics).delete()
            session.commit()
        except Exception as e:
            print(f"Error clearing metrics database: {e}")
            session.rollback()
            return {"status": "error", "msg": "An error occurred while clearing metrics database."}

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "email": self.email,
            "password": self.password,
        }
    
    def __repr__(self) -> str:
        return str(self.to_dict())

    
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


# session.add(new_comment)
# session.commit()
# session.add(new_user)
# session.commit()

def handle_error(e, msg):
    print(f"Error: {msg}: {e}")
    return {"status": "error", "msg": f"An error occurred while {msg}."}

def clear_table(session, table, msg):
    try:
        session.query(table).delete()
        session.commit()
    except Exception as e:
        session.rollback()
        return handle_error(e, f"{msg} database")




@app.route('/token', methods=["POST"])
def create_token():
    data = request.get_json()
    validate = Users.validate_user(session=session, data=data)
    if validate['status'] == "error":
        return {"msg": "Wrong email or password"}, 401

    access_token = create_access_token(identity=data['email'])
    response = {"access_token":access_token}
    return response

@app.route('/run_sentiment', methods=["GET"])
def new_sentiment():
    training_data = Comments.get_training(session)
    testing_data = Comments.get_testing(session)

    vectorizer = TfidfVectorizer()
    preprocessor = Preprocessing()
    label_mapping = {
        Sentiments.POSITIVE: 1,
        Sentiments.NEUTRAL: 0,
        Sentiments.NEGATIVE: -1
    }

    preprocessed_training = []
    for data in training_data:
        result = {
            "id": data.id,
            "result": preprocessor.preprocessing_function(data.comment)
        }
        preprocessed_training.append(result)

    preprocessed_testing = []
    for data in testing_data:
        result = {
            "id": data.id,
            "result": preprocessor.preprocessing_function(data.comment)
        }
        preprocessed_testing.append(result)

    training_labels = [label_mapping[comment.label] for comment in training_data]
    training_texts = [" ".join(result["result"]) for result in preprocessed_training]
    testing_labels = [label_mapping[comment.label] for comment in testing_data]
    testing_texts = [" ".join(result["result"]) for result in preprocessed_testing]

    X_train_tfidf = vectorizer.fit_transform(training_texts)
    information_gain = mutual_info_classif(X_train_tfidf, training_labels)
    feature_info_gain = list(zip(vectorizer.get_feature_names_out(), information_gain))
    sorted_feature_info_gain = sorted(feature_info_gain, key=lambda x: x[1], reverse=True)

    top_n = 10
    selected_features = [feature for feature, _ in sorted_feature_info_gain[:top_n]]

    vectorizer_selected = TfidfVectorizer(vocabulary=selected_features)
    # X_train_selected = vectorizer_selected.fit_transform(training_texts)
    X_train_selected = X_train_tfidf

    svm_model = SVC(kernel='rbf', decision_function_shape='ovr', random_state=42)
    svm_model.fit(X_train_selected, training_labels)

    # X_test_selected = vectorizer_selected.transform(testing_texts)
    X_test_selected = vectorizer.transform(testing_texts)
    y_pred = svm_model.predict(X_test_selected)

    accuracy = accuracy_score(testing_labels, y_pred)
    accuracy_manual = calculate_accuracy(testing_labels, y_pred)
    precision, recall, f1_score = calculate_precision_recall_f1(testing_labels, y_pred, Sentiments.POSITIVE)
    confusion = confusion_matrix(testing_labels, y_pred).tolist()

    data = [
    {
        "comment_id": comment.id,
        "comment": comment.comment,
        "movie": comment.movie,
        "classification": (
            Sentiments.POSITIVE if pred == 1 else
            Sentiments.NEGATIVE if pred == -1 else
            Sentiments.NEUTRAL
        )
    } for comment, pred in zip(testing_data, y_pred)
]
    
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score
    }

    #Save clasification & metrics results
    Classification.save_classifications(session=session, data=data)
    Metrics.update_metrics(session=session, data=metrics)

    return jsonify({
    "Accuracy Manual": accuracy_manual,
    "Precision": precision,
    "Recall": recall,
    "F1-Score": f1_score,
    "confusion": confusion,
})


@app.route("/logout", methods=["POST"])
def logout():
    response = jsonify({"msg": "logout successful"})
    unset_jwt_cookies(response)
    return response

@app.route("/movies-classified")
def get_classified_movies():
    movies_list = Movies.get_classified_movies(session)
    movies_data = []
    for movie in movies_list:
        data = {
            "id": movie.id,
            "title": movie.title,
            "year": movie.year,
            "genre": movie.genre,
            "description": movie.description,
            "cover": movie.cover,
            "trailer": movie.trailer,
            "comments": Classification.get_comments(session=session, id=movie.id)
        }
        movies_data.append(data)
    data_json = json.dumps(movies_data, default=Classification.sentiment_enum_serializer)
    return data_json

@app.route("/movies-data")
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
    data_json = json.dumps(result, default=Classification.sentiment_enum_serializer)
    if "error" in result:
        return jsonify({"error": result["error"]}), 404
    else:
        return data_json, 200
    
@app.route("/search/<string:param>", methods=["GET"])
def search_movie(param):
    movies_list = Movies.search_movie(session=session, param=param)
    movies_data = []
    for movie in movies_list:
        data = {
            "id": movie.id,
            "title": movie.title,
            "year": movie.year,
            "genre": movie.genre,
            "description": movie.description,
            "cover": movie.cover,
            "trailer": movie.trailer,
            "comments": Classification.get_comments(session=session, id=movie.id)
        }
        movies_data.append(data)
    data_json = json.dumps(movies_data, default=Classification.sentiment_enum_serializer)
    return data_json

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
    try:
        comments = Comments.get_all_comments(session)
        results = [
            {
                "id": comment.id,
                "comment": comment.comment,
                "movie": comment.movie_title.title,
                "label": comment.label,
                "category": comment.category,
            }
            for comment in comments
        ]
        data_json = json.dumps(results, default=Comments.sentiment_enum_serializer)
        return data_json
    except Exception as e:
        print(f"Error fetching comments: {e}")
        return jsonify({"status": "error", "msg": "An error occurred while fetching comments."}), 500

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
def classification():
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
    label_mapping = {
        Sentiments.POSITIVE: 1,
        Sentiments.NEUTRAL: 0,
        Sentiments.NEGATIVE: -1
    }
    labels = [label_mapping[comment.label] for comment in comments]

    # Get results from preprocessing
    texts = [" ".join(result["result"]) for result in results]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, train_size=0.8, random_state=42)

    # Start Information Gain
    X_train_tfidf = vectorizer.fit_transform(X_train)
    information_gain = mutual_info_classif(X_train_tfidf, y_train)
    feature_info_gain = list(zip(vectorizer.get_feature_names_out(), information_gain))
    sorted_feature_info_gain = sorted(feature_info_gain, key=lambda x: x[1], reverse=True)


    # Select top 10 features with highest IG
    top_n = 10
    selected_features = [feature for feature, _ in sorted_feature_info_gain[:top_n]]

    # Start TF-IDF
    vectorizer_selected = TfidfVectorizer(vocabulary=selected_features)
    X_train_selected = vectorizer_selected.fit_transform(X_train)

    # Make the model
    svm_model = SVC(kernel='rbf', decision_function_shape='ovr')
    svm_model.fit(X_train_selected, y_train)

    # Vectorize the testing data using TF-IDF with only the selected features
    X_test_selected = vectorizer_selected.transform(X_test)

    # Make the predictions
    y_pred = svm_model.predict(X_test_selected)

    # Evaluate the classification
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_manual = calculate_accuracy(y_test, y_pred)
    precision, recall, f1_score = calculate_precision_recall_f1(y_test, y_pred, Sentiments.POSITIVE)

    data = [
    {
        "comment_id": comment.id,
        "comment": comment.comment,
        "movie": comment.movie,
        "classification": (
            Sentiments.POSITIVE if pred == 1 else
            Sentiments.NEGATIVE if pred == -1 else
            Sentiments.NEUTRAL
        )
    } for comment, pred in zip(comments, y_pred)
]
    
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score
    }

    #Save clasification & metrics results
    Classification.save_classifications(session=session, data=data)
    Metrics.update_metrics(session=session, data=metrics)

    return jsonify({
    "Accuracy Manual": accuracy_manual,
    "Precision": precision,
    "Recall": recall,
    "F1-Score": f1_score,
})

@app.route("/select-classifications")
def select_classifications():
    raw = Classification.get_all_classifications(session)
    data_list = []
    for item in raw:
        data = {
            "id": item.id,
            "comment_id": item.comment_id,
            "comment": item.comment,
            "movie": item.movie_title.title,
            "classification": item.classification,
            "validation": Comments.validate_label(session=session, comment_id=item.comment_id),
        }
        data_list.append(data)
    data_json = json.dumps(data_list, default=Classification.sentiment_enum_serializer)
    return data_json

@app.route("/select-training")
def select_training():
    try:
        comments = Comments.get_training(session)
        results = [
            {
                "id": comment.id,
                "comment": comment.comment,
                "movie": comment.movie_title.title,
                "label": comment.label,
            }
            for comment in comments
        ]
        data_json = json.dumps(results, default=Comments.sentiment_enum_serializer)
        return data_json
    except Exception as e:
        print(f"Error fetching training data: {e}")
        return jsonify({"status": "error", "msg": "An error occurred while fetching training data."}), 500

@app.route("/select-testing")
def select_testing():
    try:
        comments = Comments.get_testing(session)
        results = [
            {
                "id": comment.id,
                "comment": comment.comment,
                "movie": comment.movie_title.title,
                "label": comment.label,
            }
            for comment in comments
        ]
        data_json = json.dumps(results, default=Comments.sentiment_enum_serializer)
        return data_json
    except Exception as e:
        print(f"Error fetching testing data: {e}")
        return jsonify({"status": "error", "msg": "An error occurred while fetching testing data."}), 500

@app.route("/select-metrics")
def get_metrics():
    try:
        metrics = Metrics.get_metrics(session=session)
        data = {
            "accuracy": f"{metrics.accuracy * 100:.2f}%",
            "precision": f"{metrics.precision * 100:.2f}%",
            "recall": f"{metrics.recall * 100:.2f}%",
            "f1_score": f"{metrics.f1_score * 100:.2f}%"
        }
        return jsonify(data)
    except Exception as e:
        print(f"Error fetching metrics: {e}")
        return jsonify({"status": "error", "msg": "An error occurred while fetching metrics."}), 500

@app.route("/reset-classification")
def reset_classification():
    try:
        clear_metrics = clear_table(session, Metrics, "clearing metrics")
        clear_classification = clear_table(session, Classification, "deleting classification")
        if clear_metrics is None and clear_classification is None:
            return jsonify({"msg": "Success resetting the classification"}), 200
    except Exception as e:
        return handle_error(e, "resetting classification")



if __name__ == '__main__':
    app.run(debug=True)