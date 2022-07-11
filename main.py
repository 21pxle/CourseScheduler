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
app.config.from_object('config.InitDevConfig')
db = SQLAlchemy(app)
bcrypt_flask = Bcrypt(app)


@app.before_request
def load_student():
    g.user = None
    student_id = session.get("student_id")
    if student_id:
        student = Student.query.get(student_id)
        g.user = student


# The index route would provide the "basic" functionalities of the course selection.
@app.route('/', methods=['POST', 'GET'])
def index():
    return render_template('/index.html')


@app.route('/login.html', methods=['POST', 'GET'])
def login():
    error = None
    if request.method == 'POST':
        args = request.form
        email = args.get('email')
        student = Student.query.filter(Student.email == email).first()
        hashed_password = student.password
        password = args.get('password')

        if bcrypt_flask.check_password_hash(hashed_password, password):
            session['student_id'] = student.id
            return redirect(url_for("profile"))
        else:
            error = "Incorrect email or password. Please try again."
    return render_template('/login.html', error=error)


# The login route would provide the more advanced functionalities of the course selection.
@app.route('/signup.html', methods=['POST', 'GET'])
def signup():
    error = None
    if request.method == 'POST':
        # Take all form parameters (email, password, first, last, middle)
        args = request.form
        email = args.get('email')
        password = args.get('password')
        repeat_password = args.get('repeat_password')
        hashed_password = bcrypt_flask.generate_password_hash(password)
        first = args.get('first')
        last = args.get('last')
        middle = args.get('middle')
        existing_email = Student.query.filter(Student.email == email).first()
        if existing_email:
            error = "Email has already been taken."
        elif repeat_password != password:
            error = "Passwords must match."
        else:
            student = Student(email=email, password=hashed_password, first_name=first,
                              last_name=last, middle_initial=middle, major="Undecided")
            db.session.add(student)
            db.session.commit()
            session['student_id'] = student.id

            return redirect(url_for("profile"))
    return render_template('/signup.html', error=error)


# Dict, sorted from Monday to Friday, Start to End, with respect to X.
def get_intervals(preferences):
    keys = list(preferences.keys())
    result = {'monday': [], 'tuesday': [], 'wednesday': [], 'thursday': [], 'friday': []}
    days = [key for key in result.keys()]
    day = 0
    # key: a string that has a day and a start/end trigger.
    for i in [x for x in range(len(keys)) if x % 2 == 0]:
        if not keys[i].startswith(days[day]):
            day += 1
        start = preferences[keys[i]]
        end = preferences[keys[i + 1]]
        if len(start.split(':')) == 2:
            start += ':00'
        if len(end.split(':')) == 2:
            end += ':00'
        result[days[day]].append([start, end])
    for d in days:
        # Sort by start time (the first time listed is the start time).
        result[d] = sorted(result[d], key=lambda x: get_time(x[0]))
    return result


def get_time(time):
    hours, minutes, secs = time.split(':')
    return 60 * int(hours) + int(minutes)


@app.route('/profile', methods=['POST', 'GET'])
def profile():
    # Preferences
    error = None
    if not g.user:
        return redirect(url_for("login"))

    if request.method == "POST":
        # TODO add preferences for the user.
        # Validate preferences
        form = request.form
        """db.session.add(AvailableTimes(
            student_id=g.user.id,
            day=1,
            start='2:00',
            end='3'
        ))"""

        # There should be no overlapping intervals. The intervals variable is designed to hold 2-tuples of strings.
        intervals = get_intervals(form)
        overlap = False
        for key in intervals:
            for i in range(len(intervals[key]) - 1):
                # An overlapping interval starts later than the earlier interval ends.
                if intervals[key][i + 1][0] < intervals[key][i][1]:
                    overlap = True
                    break
            else:
                continue
            break
        if not overlap:
            day_index = {'monday': 1, 'tuesday': 2, 'wednesday': 3, 'thursday': 4, 'friday': 5}
            AvailableTimes.query.filter_by(student_id=g.user.id).delete()
            for day in intervals:
                for interval in intervals[day]:
                    preference = AvailableTimes(student_id=g.user.id, day=day_index[day],
                                                start=interval[0], end=interval[1])
                    db.session.add(preference)

            db.session.commit()
            pass
        else:
            error = "Time intervals cannot overlap each other."

    # range(1, 6) refers to every number from 1 to 5, inclusive.
    prefs = [AvailableTimes.query.filter(AvailableTimes.student_id == g.user.id,
                                         AvailableTimes.day == x).all() for x in range(1, 6)]
    prefs = dict(enumerate(prefs, start=1))
    return render_template('/profile.html', user=g.user, prefs=prefs, error=error)


# The classes corresponding to the tables in the database are shown below.
# Notice that each field of the class is a column that contains a data type.

# db.relationship('User', foreign_keys=user_id)
class Course(db.Model):
    """Data model for courses"""

    __tablename__ = 'Course'

    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), nullable=False)
    abbreviation = db.Column(db.String(10), nullable=False)
    category = db.Column(db.String(10), nullable=False)


class Corequisite(db.Model):
    """Data model for co-requisite courses (to be taken with the course)."""

    __tablename__ = 'Corequisite'

    id = db.Column(db.Integer, primary_key=True)
    course_id = db.Column(db.Integer, db.ForeignKey('Course.id'))
    course = db.relationship('Course', foreign_keys=course_id,
                             primaryjoin="Course.id == Corequisite.course_id")
    corequisite_id = db.Column(db.Integer, db.ForeignKey('Course.id'))
    corequisite = db.relationship('Course', foreign_keys=corequisite_id,
                                  primaryjoin="Course.id == Corequisite.corequisite_id")
    alternate_number = db.Column(db.Integer, nullable=False)


class Prerequisite(db.Model):
    """Data model for prerequisite courses."""

    __tablename__ = 'Prerequisite'

    id = db.Column(db.Integer, primary_key=True)
    course_id = db.Column(db.Integer, db.ForeignKey('Course.id'), nullable=False)
    course = db.relationship('Course', foreign_keys=course_id)
    prerequisite_id = db.Column(db.Integer, db.ForeignKey('Course.id'), nullable=False)
    prerequisite = db.relationship('Course', foreign_keys=prerequisite_id,
                                   primaryjoin="Course.id == Prerequisite.prerequisite_id")
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
    """Data model for professor reviews."""

    __tablename__ = 'Review'

    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('Student.id'), nullable=False)
    student = db.relationship('Student', foreign_keys=student_id)
    content = db.Column(db.String(1000))
    rating = db.Column(db.DECIMAL(5, 4))
    alternate_number = db.Column(db.Integer, nullable=False)


class Section(db.Model):
    """Some info about the coursework"""

    __tablename__ = 'Section'

    id = db.Column(db.Integer, primary_key=True)
    course_id = db.Column(db.Integer, db.ForeignKey('Course.id'), nullable=False)
    course = db.relationship('Course', foreign_keys=course_id)
    student_id = db.Column(db.Integer, db.ForeignKey('Student.id'), nullable=False)
    student = db.relationship('Student', foreign_keys=student_id)
    professor_id = db.Column(db.Integer, db.ForeignKey('Professor.id'), nullable=False)
    professor = db.relationship('Professor', foreign_keys=professor_id)
    grade = db.Column(db.String(2))
    min_pass_grade = db.Column(db.String(2))
    seats = db.Column(db.Integer, nullable=False)


class Schedule(db.Model):
    """Data model for the scheduling of the courses"""

    __tablename__ = "Schedule"

    id = db.Column(db.Integer, primary_key=True)
    section_id = db.Column(db.Integer, db.ForeignKey('Section.id'))
    section = db.relationship('Section', foreign_keys=section_id)
    mondays = db.Column(db.Boolean, nullable=False)
    tuesdays = db.Column(db.Boolean, nullable=False)
    wednesdays = db.Column(db.Boolean, nullable=False)
    thursdays = db.Column(db.Boolean, nullable=False)
    fridays = db.Column(db.Boolean, nullable=False)
    start = db.Column(db.Time, nullable=False)
    end = db.Column(db.Time, nullable=False)


class AvailableTimes(db.Model):
    """Data model for student availablity"""

    __tablename__ = "AvailableTimes"

    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('Student.id'))
    student = db.relationship('Student', foreign_keys=student_id)
    day = db.Column(db.Integer, nullable=False)
    start = db.Column(db.Time, nullable=False)
    end = db.Column(db.Time, nullable=False)


db.session.execute("CREATE DATABASE IF NOT EXISTS courses;")
app.config.from_object('config.DevConfig')
db.create_all()

if __name__ == '__main__':
    app.run()
