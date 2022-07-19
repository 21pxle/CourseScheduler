# import csv
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
from datetime import date, datetime, timedelta
from flask_bcrypt import Bcrypt
from flask_sqlalchemy import SQLAlchemy

# import os.path

app = Flask(__name__)
app.config.from_object('config.InitDevConfig')
db = SQLAlchemy(app)
bcrypt_flask = Bcrypt(app)
page_num = 0  # For results
results, courses = [], []


@app.before_first_request
def permanent_session():
    session.permanent = True
    app.permanent_session_lifetime = timedelta(minutes=20)


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
    global page_num, results, courses
    if request.method == 'POST':
        if 'prefs' in request.form:
            return redirect(url_for("profile"))
        if 'courses' in request.form:
            courses = request.form.getlist('courses')
            if g.user is None:
                results = guest_algorithm(courses)
            else:
                results = genetic_algorithm(courses, 0.8)
        if 'page' in request.form:
            if 'previous' in request.form:
                page_num = max(0, page_num - 1)
            elif 'next' in request.form:
                page_num = min((len(results) - 1) // 25, page_num + 1)
            else:
                page_num = int(request.form['nav']) - 1

    selected = [x for (x,) in Course.query.with_entities(Course.title)
                                    .filter(Course.id.in_(courses)).order_by(Course.id.asc()).all()]

    return render_template('/index.html', courses=all_courses, selected_courses=selected,
                           query=display(results, page_num), page=page_num, end=min(25 * (page_num + 1), len(results)),
                           num_pages=(len(results) + 24) // 25, results=results, user=g.user)


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
        form = request.form
        if "scheduler" in form:
            return redirect(url_for("index"))
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
    title = db.Column(db.String(100), unique=True, nullable=False)
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

    def times(self):
        scheduled = [self.mondays, self.tuesdays, self.wednesdays, self.thursdays, self.fridays]
        text = ["mondays", "tuesdays", "wednesdays", "thursdays", "fridays"]
        days = [word.capitalize() for (idx, word) in enumerate(text) if scheduled[idx]]
        today = date.today()
        scheduled_days = ", ".join(days[:(len(days) - 1)]) + " and " + days[len(days) - 1] if len(days) > 1 else days[0]
        scheduled_times = f"{datetime.combine(today, self.start).strftime('%I:%M %p')} to " \
                          f"{datetime.combine(today, self.end).strftime('%I:%M %p')}"
        return scheduled_days + ", " + scheduled_times


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
all_professors = Professor.query.all()

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
        lst = list(set(rules[i]) - {genome[j]})
        if len(lst):
            genome[j] = random.choice(lst)


def guest_algorithm(selected_courses):
    result, rules = [], [[x for (x,) in Schedule.query.with_entities(Schedule.id).filter(
        Schedule.course_id == course).all()] for course in selected_courses]

    result = set()
    genomes = [[]]
    log_product = sum(math.log10(len(rule)) for rule in rules)

    # Count all population if it is less than 50k. The logarithm of that number happens to be 5 - log10(2).
    if log_product < 5 - math.log10(2):
        for count, r in enumerate(rules):
            genomes = itertools.product(genomes, r)
            genomes = list(flatten(genomes))
            genomes = [genome for genome in chunks(genomes, count + 1) if overlap_test(genome)]

        result = random.sample(genomes, k=min(1000, len(genomes)))

        return [(Schedule.query.filter(Schedule.id.in_(genome)).order_by(Schedule.id.asc()).all(), 100)
                for genome in result]

    trials = 0
    while len(result) < 1000 and trials < 50000:
        genome = [random.choice(rule) for rule in rules]
        if overlap_test(genome):
            result.add((*genome,))
        trials += 1

    return [(Schedule.query.filter(Schedule.id.in_(genome)).order_by(Schedule.id.asc()).all(), 100) for genome in result]


def display(query, page=0):
    return query[(25 * page):(25 * (page + 1))]


def flatten(lst):
    for i in lst:
        yield from [i] if not isinstance(i, tuple) and not isinstance(i, list) else flatten(i)


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


# Genome: A list of numbers that concern the course IDs.
# Gene: The ID of that specific course
def brute_force(lst):
    # The brute-force approach is the most straightforward.
    # It is used when there is a sufficiently small number
    # of combinations of courses (order less than ~50k).
    lst_courses = [Schedule.query.filter(Schedule.course_id == course_id).all() for course_id in lst]
    population = list(itertools.product(*lst_courses))

    # This parameter should change according to the preferences.
    # For more details, please refer to the fitness function.
    scores = [fitness(genome, 0.8) for genome in population]
    possibilities = list(filter(lambda x: x[1] > 0, list(zip(population, scores))))
    possibilities = sorted(possibilities, key=lambda x: x[1], reverse=True)

    # Take the top 5 solutions.
    return possibilities[:5]


# Rules: A list of numbers that govern the number of sections with the specific course ID.


def crossover(genome1, genome2):
    assert len(genome1) == len(genome2), "Lengths are not equal."
    point = random.randint(1, len(genome1) - 1)
    return genome1[:point] + genome2[point:], genome2[:point] + genome1[point:]


def overlap_test(genome):
    intervals = {'monday': [], 'tuesday': [], 'wednesday': [], 'thursday': [], 'friday': []}
    days = [key for key in intervals.keys()]
    professors = []

    # Every gene will have their intervals added to the dict shown above.
    schedules = Schedule.query.filter(Schedule.id.in_(genome)).all()
    if not len(schedules):
        raise ValueError('No course section found. Please try again.')
    for schedule in schedules:
        professors.append(schedule.professor_id)
        scheduled_days = [schedule.mondays, schedule.tuesdays, schedule.wednesdays,
                          schedule.thursdays, schedule.fridays]
        scheduled_times = [schedule.start, schedule.end]
        for day_num, scheduled in enumerate(scheduled_days):
            if scheduled:
                intervals[days[day_num]].append(scheduled_times)

    for day in intervals:
        intervals[day] = sorted(intervals[day], key=lambda x: x[0])

    # Check for overlaps in break time.
    for key in intervals:
        for i in range(len(intervals[key]) - 1):
            # An overlapping interval starts later than the earlier interval ends.
            if intervals[key][i + 1][0] < intervals[key][i][1]:
                return False
    return True


def relative_rating(genome, professors):
    global all_ratings
    # Standardize ratings only if there is a difference between the ratings:
    ratings_dict = dict(all_ratings)
    score = 0
    for prof_id in professors:
        score += ratings_dict[prof_id]
    score *= 100 / len(genome)
    return score


def fitness(genome, alpha):
    # Each gene represents one section ID. The availability of a person is a dict of intervals, which each consist of
    # a start time and an end time. It also requires the user to be registered.
    intervals = {'monday': [], 'tuesday': [], 'wednesday': [], 'thursday': [], 'friday': []}
    days = [key for key in intervals.keys()]
    professors = []
    # Special cases of alpha
    # Every gene will have their intervals added to the dict shown above.
    schedules = Schedule.query.filter(Schedule.id.in_(genome)).all()

    course_time = timedelta(0)
    today = date.today()
    for schedule in schedules:
        professors.append(schedule.professor_id)
        scheduled_days = [schedule.mondays, schedule.tuesdays, schedule.wednesdays,
                          schedule.thursdays, schedule.fridays]
        course_time += (datetime.combine(today, schedule.end) -
                        datetime.combine(today, schedule.start)) * sum(scheduled_days)
        for day_num, scheduled in enumerate(scheduled_days):
            if scheduled:
                intervals[days[day_num]].append([schedule.start, 'S'])  # Mark as start of interval
                intervals[days[day_num]].append([schedule.end, 'E'])  # Mark as end of interval

    if not len(schedules):
        raise TypeError('No course section found. Please try again.')
    intervals = {day: sorted(intervals[day], key=lambda x: x[0]) for day in intervals}
    # Check for overlaps in break time.
    for key in intervals:
        count = 0
        for idx, interval in enumerate(intervals[key]):
            # An overlapping interval starts later than the earlier interval ends.
            count += 1 if interval[1] == "S" else -1
            if count > 1:
                return 0
    if alpha == 0:
        return relative_rating(genome, professors)

    available_intervals = [AvailableTimes.query.filter(AvailableTimes.student_id == g.user.id,
                                                       AvailableTimes.day == x).all() for x in range(1, 6)]
    for i in range(5):
        for interval in available_intervals[i]:
            intervals[days[i]].append([interval.start, "S"])
            intervals[days[i]].append([interval.end, "E"])

    for day in intervals:
        intervals[day] = sorted(intervals[day], key=lambda x: x[0])
    # Availability overlap should be maximized.
    overlap = timedelta(0)

    count = 0

    for key in intervals:
        for idx, interval in enumerate(intervals[key]):
            # An overlapping interval starts later than the earlier interval ends.
            if count > 1:
                overlap += datetime.combine(today, interval[0]) \
                           - datetime.combine(today, intervals[key][idx - 1][0])
            count += 1 if interval[1] == "S" else -1

    # Short-circuiting operation since it's moot that ratings are unnecessary.
    if alpha == 1:
        return 100 * overlap / course_time

    return 100 * alpha * (overlap / course_time) + (1 - alpha) * relative_rating(genome, professors)


def genetic_algorithm(selected_courses, alpha):
    # Alpha is a parameter that takes into account the preferences of a better professor
    # vs. a timeline more flexible. The courses are based on the user's preferences.
    # If the ratings don't account in the decision to take some course,
    # the alpha value should be set to 1.0. Of course, if only ratings
    # matter then the alpha value should be set to 0. By default, alpha is 0.8,
    # which means that the program will prioritize availability constraints over
    # better professor ratings.

    # For smaller problem sizes, brute forcing would work better than genetic algorithms
    # because the speed advantage is less than that of brute force.

    global all_ratings

    result, rules = [], [[x for (x,) in
                          Schedule.query.with_entities(Schedule.id).filter(Schedule.course_id == course).all()]
                         for course in selected_courses]

    ratings = Review.query.with_entities(Review.professor_id, db.func.avg(Review.rating).label('average'),
                                         db.func.count(Review.rating).label('count')).group_by(
        Review.professor_id).all()
    ratings = [(prof_id, float(rating), count) for (prof_id, rating, count) in ratings]
    prof_adj_ratings = [
        (prof_id, rating, (rating * count + 9.604 - 1.96 * math.sqrt(count * rating * (5 - rating) + 24.01))
         / (count + 3.8416)) for (prof_id, rating, count) in ratings]
    all_ratings = [(prof_id, round(adj_rating, 3) + 0.0001 * round(rating, 3))
                   for (prof_id, rating, adj_rating) in prof_adj_ratings]

    ratings_list = [rating for (_, rating) in all_ratings]
    r = max(ratings_list) - min(ratings_list)
    if r > 0:
        ratings_list = (np.array(ratings_list) - min(*ratings_list)) / r  # Normalize between 0 and 1.
        all_ratings = [(x, y) for (x, y) in zip(all_professors, ratings_list)]
    else:
        all_ratings = [(prof.id, 1) for prof in all_professors]

    population = [[random.choice(rule) for rule in rules] for _ in range(500)]
    for i in range(100):
        scores = [fitness(genome, alpha) for genome in population]

        # Fitness score will be between 0 and 100.
        scoreboard = {tuple(k): v for (k, v) in sorted(list(zip(population, scores)), key=lambda x: x[1], reverse=True)}
        population, scores = [list(k) for k in scoreboard.keys()], list(scoreboard.values())
        print(f'Iteration {i}, {scoreboard[next(iter(scoreboard))]}')
        # Prematurely end the loop if the top result has a 98% or higher match rating.
        if scoreboard[next(iter(scoreboard))] >= 98:
            break

        # Cast into variables from 0 to 1.
        scores = np.array(scores)
        scores -= min(scores)
        scores /= max(scores)
        scores += 0.1
        # The number of individuals should be 500 for the next generation.
        # Due to the uniqueness of the population, however, it should be noted
        # that if there are less than 100 individuals, then there must be more
        # offspring to compensate for the lack of diversity.
        fathers = random.choices(population=population, weights=scores, k=max(200, 250 - len(population) // 2))
        mothers = random.choices(population=population, weights=scores, k=max(200, 250 - len(population) // 2))
        offspring = [crossover(mother, father) for father, mother in zip(fathers, mothers)]
        offspring = [x for xs in offspring for x in xs]  # Unpack a list of lists

        for genome in offspring:
            # There is a 1% chance that a specific gene will be mutated.
            num_genes_mutated = np.random.binomial(7, 0.01)
            genes_mutated = random.sample(list(range(len(selected_courses))), k=num_genes_mutated)
            if num_genes_mutated:
                mutate(genome, genes_mutated, [rules[gene] for gene in genes_mutated])

        # Top 100 solutions also get into the next generation
        population = [list(x) for x in list(scoreboard)[:100] + offspring]
        if len(population) > 500:
            population.pop()
    # Return the best fits if they are close enough to the solution.
    scores = [fitness(genome, alpha) for genome in population]
    scoreboard = sorted(list(zip([tuple(genome) for genome in population], scores)), key=lambda x: x[1], reverse=True)
    max_score = scoreboard[0][1]
    result = dict(itertools.takewhile(lambda x: x[1] >= 0.95 * max_score, scoreboard))
    return [(genome, round(score, 1)) for genome, score in zip([Schedule.query.filter(Schedule.id.in_(genome))
                                                               .order_by(Schedule.id.asc()).all()
                                                                for genome in result.keys()], list(result.values()))]


all_ratings = []

if __name__ == '__main__':
    app.run()
