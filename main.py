import csv
import random

import math
import itertools
import numpy as np
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
    return render_template('/index.html', courses=all_courses)


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
        result[days[day]].append([start, end])

    for d in days:
        # Sort by start time (the first time listed is the start time).
        result[d] = sorted(result[d], key=lambda x: get_time(x[0]))

    return result


def get_time(time):
    hours, minutes = time.split(':')[:2]
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
                if get_time(intervals[key][i + 1][0]) < get_time(intervals[key][i][1]):
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
    professor_id = db.Column(db.Integer, db.ForeignKey('Professor.id'), nullable=False)
    professor = db.relationship('Professor', foreign_keys=professor_id)
    content = db.Column(db.String(1000))
    rating = db.Column(db.DECIMAL(5, 4))  # I don't know why this must be in all caps.


class Schedule(db.Model):
    """Some info about the coursework"""

    __tablename__ = 'Schedule'

    id = db.Column(db.Integer, primary_key=True)
    section = db.Column(db.String(5), nullable=False)
    course_id = db.Column(db.Integer, db.ForeignKey('Course.id'), nullable=False)
    course = db.relationship('Course', foreign_keys=course_id)
    professor_id = db.Column(db.Integer, db.ForeignKey('Professor.id'), nullable=False)
    professor = db.relationship('Professor', foreign_keys=professor_id)
    min_pass_grade = db.Column(db.String(2))
    start = db.Column(db.Time, nullable=False)
    end = db.Column(db.Time, nullable=False)
    seats = db.Column(db.Integer, nullable=False)
    mondays = db.Column(db.Boolean, nullable=False)
    tuesdays = db.Column(db.Boolean, nullable=False)
    wednesdays = db.Column(db.Boolean, nullable=False)
    thursdays = db.Column(db.Boolean, nullable=False)
    fridays = db.Column(db.Boolean, nullable=False)


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
db.session.execute("USE courses;")
db.create_all()
all_courses = Course.query.all()

'''
with open('static/data/Professor.csv', 'r') as file:
    for row in csv.DictReader(file):
        db.session.add(Professor(last_name=row['Last'], first_name=row['First'], middle_initial=row['Middle']))

db.session.commit()

with open('static/data/Review.csv', 'r') as file:
    for row in csv.DictReader(file):
        db.session.add(Review(student_id=int(row['Student ID']), professor_id=int(row['Professor ID']),
                              content=row['Content'], rating=row['Rating']))

db.session.commit()
with open('static/data/Courses.csv', 'r') as file:
    for row in csv.DictReader(file):
        db.session.add(Course(title=row['Title'], category=row['Category'], abbreviation=row['Abbreviation']))
        
db.session.commit()

with open('static/data/Schedule.csv', 'r') as file:
    for row in csv.DictReader(file):
        db.session.add(Schedule(course_id=row['Course ID'], professor_id=row['Professor ID'], section=row['Section'],
                                seats=int(row['Seats']), start=row['Start'], end=row['End'],
                                mondays=int(row['Mondays']), tuesdays=int(row['Tuesdays']),
                                wednesdays=int(row['Wednesdays']), thursdays=int(row['Thursdays']),
                                fridays=int(row['Fridays'])))

db.session.commit()'''


# If you've made it this far into the code, congratulations.
# Genetic Algorithms
def mutate(genome, indices, rules):
    for i, j in enumerate(indices):
        genome[j] = random.choice(rules[i])
    return genome


# Genome: A list of numbers that concern the course IDs.
# Gene: The ID of that specific course
def brute_force(courses):
    # The brute-force approach is the most straightforward.
    # It is used when there is a sufficiently small number
    # of combinations of courses (order less than ~500k).
    courses = [Schedule.query.filter(Schedule.course_id == course_id).all() for course_id in courses]
    population = list(itertools.product(*courses))

    # This parameter should change according to the preferences.
    # For more details, please refer to the fitness function.
    scores = [fitness(genome, courses, 0.8) for genome in population]
    possibilities = list(filter(lambda x: x[1] > 0, list(zip(population, scores))))
    possibilities = sorted(possibilities, key=lambda x: x[1], reverse=True)

    # Take the top 5 solutions.
    return possibilities[:5]
# Rules: A list of numbers that govern the number of sections with the specific course ID.


def crossover(genome1, genome2):
    assert len(genome1) == len(genome2), "Lengths are not equal."
    point = random.randint(1, len(genome1) - 1)
    return genome1[:point] + genome2[point:], genome2[:point] + genome1[point:]


def genetic_algorithm(courses, alpha):
    # Alpha is a parameter that takes into account the preferences of a better professor
    # vs. a timeline more flexible. The courses are based on the user's preferences.
    # If the ratings don't account in the decision to take some course,
    # the alpha value should be set to 1.0. Of course, if only ratings
    # matter then the alpha value should be set to 0. By default, alpha is 0.8,
    # which means that the program will prioritize availability constraints over
    # better professor ratings.

    # For smaller problem sizes, brute forcing would work better than genetic algorithms
    # because the speed advantage is less than that of brute force.

    result, rules = [], [Schedule.query.filter(Schedule.course_id == course_id).all() for course_id in courses]

    population = [[random.choice(rule) for rule in rules] for _ in range(500)]
    for i in range(300):
        scores = [fitness(genome, courses, alpha) for genome in population]

        # Fitness score will be between 0 and 100.
        scoreboard = sorted(list(zip(population, scores)), key=lambda x: x[1], reverse=True)

        # Prematurely add the top 5 solutions if the fitness function is within 1% of optimal solution.
        if scoreboard[0][1] > 99:
            return [genome for (genome, score) in population[:5] if score > 95]

        # Cast into
        scores = np.array(scores)
        scores -= min(scores)
        scores /= max(scores)
        fathers = random.choices(population=population, weights=scores, k=200)
        mothers = random.choices(population=population, weights=scores, k=200)
        offspring = [crossover(father, mother) for father, mother in (fathers, mothers)]
        offspring = [x for xs in offspring for x in xs]  # Unpack a list of lists

        for genome in offspring:
            # There is a 1% chance that a specific gene will be mutated.
            num_genes_mutated = min(np.random.binomial(7, 0.01), 7)
            genes_mutated = random.sample(list(range(len(courses))), k=num_genes_mutated)
            if num_genes_mutated:
                mutate(genome, genes_mutated, [rules[gene] for gene in genes_mutated])

        # Top 100 solutions also get into the next generation
        population = [genome for genome, _ in scoreboard[:100]] + offspring
    # Return the best 5 fits if they are close enough to the solution.
    scores = [fitness(genome, courses, alpha) for genome in population]
    scoreboard = sorted(list(zip(population, scores)), key=lambda x: x[1], reverse=True)
    max_score = scoreboard[0][1]
    return [genome for genome, score in scoreboard[:5] if score > 0.95 * max_score]


def fitness(genome, courses, alpha):
    # Each gene represents one section ID. The availability of a person is a dict of intervals, which each consist of
    # a start time and an end time.
    intervals = {'monday': [], 'tuesday': [], 'wednesday': [], 'thursday': [], 'friday': []}
    days = [key for key in intervals.keys()]
    professors = []

    # Special cases of alpha
    if alpha == 0:
        adj_rating, overall_rating = 0, 0
        for prof_id in professors:

            rating = Review.query.with_entities(db.func.avg(Review.rating)
                                                .label('average')).filter(Review.professor_id == prof_id)
            count = Review.query.with_entities(db.func.count(Review.rating)
                                               .label('count')).filter(Review.professor_id == prof_id)
            if count > 0:
                overall_rating += rating
                adj_rating += (rating * count + 9.604 - 1.96 *
                               math.sqrt(count * rating * (5 - rating) + 24.01)) / (count + 3.8416)
        adj_rating = max(0, adj_rating) / len(courses)
        overall_rating = max(0, overall_rating) / len(courses)
        return 20 * adj_rating + 0.002 * overall_rating

    # Every gene will have their intervals added to the dict shown above.
    for i, gene in enumerate(genome):
        schedule = Schedule.query.filter(Schedule.id == gene).first()
        if not schedule:
            raise TypeError('No course section found. Please try again.')
        professors.append(schedule.professor_id)
        scheduled_days = [schedule.mondays, schedule.tuesdays, schedule.wednesdays,
                          schedule.thursdays, schedule.fridays]
        scheduled_times = [get_time(schedule.start), get_time(schedule.end)]
        for day_num, scheduled in enumerate(scheduled_days):
            if scheduled:
                intervals[days[day_num]].append(scheduled_times)

    for day in intervals:
        intervals[day] = sorted(intervals[day], key=lambda x: x[0])

    # Check for overlaps in break time.
    overlap = False
    course_time = 0
    for key in intervals:
        for i in range(len(intervals[key]) - 1):
            # An overlapping interval starts later than the earlier interval ends.
            course_time += intervals[key][i][0] - intervals[key][i][0]
            if intervals[key][i + 1][0] < intervals[key][i][1]:
                overlap = True
                break
        else:
            interval = intervals[key][len(intervals[key])]
            course_time += interval[1] - interval[0]
            continue
        break

    if overlap:
        return 0  # Courses cannot overlap with each other.

    available_intervals = [AvailableTimes.query.filter(AvailableTimes.student_id == g.user.id,
                                                       AvailableTimes.day == x).all() for x in range(1, 6)]
    for i in range(5):
        for interval in available_intervals[i]:
            intervals[days[i]].append(interval)

    # Availability overlap should be maximized.
    overlap = 0.

    for key in intervals:
        for i in range(len(intervals[key]) - 1):
            if intervals[key][i + 1][0] < intervals[key][i][1]:
                overlap += intervals[key][i][1] - intervals[key][i + 1][0]

    # Short-circuiting operation since it's moot that ratings are unnecessary.
    if alpha == 1:
        return 100 * overlap / course_time

    # Add ratings:
    adj_rating, overall_rating = 0, 0
    for prof_id in professors:

        rating = Review.query.with_entities(db.func.avg(Review.rating)
                                            .label('average')).filter(Review.professor_id == prof_id)
        count = Review.query.with_entities(db.func.count(Review.rating)
                                           .label('count')).filter(Review.professor_id == prof_id)
        if count > 0:
            overall_rating += rating
            adj_rating += (rating * count + 9.604 - 1.96 *
                           math.sqrt(count * rating * (5 - rating) + 24.01)) / (count + 3.8416)
    adj_rating = max(0, adj_rating) / len(courses)
    overall_rating = max(0, overall_rating) / len(courses)

    return 100 * alpha * (overlap / course_time) + (1 - alpha) * (20 * adj_rating + 0.002 * overall_rating)


if __name__ == '__main__':
    app.run()
