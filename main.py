from datetime import datetime

import sqlalchemy
from flask import (
    Flask,
    render_template,
    request,
    url_for,
    redirect,
    g,
    session)
from flask_bcrypt import Bcrypt
from flask_sqlalchemy import SQLAlchemy

# import os.path

app = Flask(__name__)
app.config.from_object('config.DevConfig')
db = SQLAlchemy(app)


bcrypt_flask = Bcrypt(app)

# Initialize the student planner.
for line in open('StudentPlanner.sql'):
    db.session.execute(line)


# The index route would provide the "basic" functionalities of the course selection.
@app.route('/', methods=['POST', 'GET'])
def index():
    return render_template('/index.html')


@app.route('/login.html', methods=['POST', 'GET'])
def login():
    return render_template('/login.html')


# The login route would provide the more advanced functionalities of the course selection.
@app.route('/signup.html', methods=['POST', 'GET'])
def signup():
    args = request.args
    email = args.get('email')
    if email:
        existing_user = Student.query.filter()
    return render_template('/signup.html')


@app.route('/profile/<name>', methods=['POST', 'GET'])
def profile(name):
    return render_template('/profile.html', name=name)


# The classes corresponding to the tables in the database are shown below.
# Notice that each field of the class is a column that contains a data type.

# db.relationship('User', foreign_keys=user_id)
class Course(db.Model):
    """Data model for students"""

    __tablename__ = 'Course'

    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), nullable=False)
    password = db.Column(db.String(60), nullable=False)
    last_name = db.Column(db.String(40), nullable=False)
    first_name = db.Column(db.String(20), nullable=False)
    middle_initial = db.Column(db.String(1))
    major = db.Column(db.String(50), nullable=False)

    def __repr__(self):
        return '{} {}'.format(self.first_name, self.last_name)


class Corequisite(db.Model):
    """Data model for co-requisite courses (to be taken with the course)."""

    __tablename__ = 'Corequisite'

    id = db.Column(db.Integer, primary_key=True)
    course_id = db.relationship('Course', foreign_keys=id)
    course = db.Column(db.Integer, db.ForeignKey('Course.id'))
    corequisite_id = db.relationship('Course', foreign_keys=id)
    corequisite = db.Column(db.Integer, db.ForeignKey('Course.id'))
    alternate_number = db.Column(db.Integer, nullable=False)


class Prerequisite(db.Model):
    """Data model for prerequisite courses."""

    __tablename__ = 'Prerequisite'

    id = db.Column(db.Integer, primary_key=True)
    course_id = db.relationship('Course', foreign_keys=id)
    course = db.Column(db.Integer, db.ForeignKey('Course.id'))
    prerequisite_id = db.relationship('Course', foreign_keys=id)
    prerequisite = db.Column(db.Integer, db.ForeignKey('Course.id'))
    alternate_number = db.Column(db.Integer, nullable=False)


class Professor(db.Model):
    """Data model for professors"""

    __tablename__ = 'Professor'

    id = db.Column(db.Integer, primary_key=True)
    last_name = db.Column(db.String(40), nullable=False)
    first_name = db.Column(db.String(20), nullable=False)
    middle_initial = db.Column(db.String(1))


class Student(db.Model):
    """Data model for students"""

    __tablename__ = 'Student'

    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(254), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)
    last_name = db.Column(db.String(40), nullable=False)
    first_name = db.Column(db.String(20), nullable=False)
    middle_initial = db.Column(db.String(1))
    major = db.Column(db.String(50), nullable=False)

    def __repr__(self):
        return '{} {}'.format(self.first_name, self.last_name)


class Review(db.Model):
    """Data model for professors"""

    __tablename__ = 'Review'

    id = db.Column(db.Integer, primary_key=True)
    student_id = db.relationship('Student', foreign_keys=id)
    student = db.Column(db.Integer, db.ForeignKey('Student.id'))
    content = db.Column(db.String(1000))
    rating = db.Column(db.DECIMAL(5, 4))
    alternate_number = db.Column(db.Integer, nullable=False)


class Section(db.Model):
    """Some info about the coursework"""

    __tablename__ = 'Section'

    id = db.Column(db.Integer, primary_key=True)
    course_id = db.relationship('Course', foreign_keys=id)
    course = db.Column(db.Integer, db.ForeignKey('Course.id'))
    student_id = db.relationship('Student', foreign_keys=id)
    student = db.Column(db.Integer, db.ForeignKey('Student.id'))
    professor_id = db.relationship('Professor', foreign_keys=id)
    professor = db.Column(db.Integer, db.ForeignKey('Professor.id'))
    grade = db.Column(db.String(2))
    min_pass_grade = db.Column(db.String(2))
    seats = db.Column(db.Integer, nullable=False)


class Schedule(db.Model):
    """Data model for the scheduling of the courses"""

    __tablename__ = "Schedule"

    id = db.Column(db.Integer, primary_key=True)
    section_id = db.relationship('Section', foreign_keys=id)
    section = db.Column(db.Integer, db.ForeignKey('Section.id'))
    mondays = db.Column(db.Boolean, nullable=False)
    tuesdays = db.Column(db.Boolean, nullable=False)
    wednesdays = db.Column(db.Boolean, nullable=False)
    thursdays = db.Column(db.Boolean, nullable=False)
    fridays = db.Column(db.Boolean, nullable=False)
    start = db.Column(db.Time, nullable=False)
    end = db.Column(db.Time, nullable=False)


class BreakTime(db.Model):
    """Data model for student preferred break times"""

    __tablename__ = "BreakTime"

    id = db.Column(db.Integer, primary_key=True)
    student_id = db.relationship('Student', foreign_keys=id)
    student = db.Column(db.Integer, db.ForeignKey('Student.id'))
    day = db.Column(db.String(1))
    start = db.Column(db.Time, nullable=False)
    end = db.Column(db.Time, nullable=False)


if __name__ == '__main__':
    app.run()
