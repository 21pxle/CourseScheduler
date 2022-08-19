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
# import timeit

# import os.path

app = Flask(__name__)
app.config.from_object('config.InitDevConfig')
db = SQLAlchemy(app)
bcrypt_flask = Bcrypt(app)
page_num = 0  # For results
results, courses = [], []


@app.before_first_request
def load_session():
    """Instantiate all variables relating to the tables of courses, sections, and professors."""
    session.permanent = True
    app.permanent_session_lifetime = timedelta(minutes=10)

    global all_schedules, all_sections, all_ratings, professors_by_course, section_ratings, schedules_by_section
    all_sections = {section.id: section for section in Section.query.all()}
    all_schedules = Schedule.query.all()
    schedules_by_section = {all_sections[key]: list(value) for (key, value) in
                            itertools.groupby(sorted(all_schedules, key=lambda schedule: schedule.id),
                                              key=lambda schedule: schedule.section_id)}
    all_schedules = {key: list(value) for (key, value) in
                     itertools.groupby(sorted(all_schedules, key=lambda schedule: schedule.id),
                                       key=lambda schedule: schedule.section_id)}  # Group all schedules by section ID.

    ratings = Review.query.with_entities(Review.professor_id, db.func.avg(Review.rating).label('average'),
                                         db.func.count(Review.rating).label('count')).group_by(
        Review.professor_id).all()
    all_ratings = dict.fromkeys([prof for prof in all_professors.keys()], 0)

    ratings = [[prof_id, float(rating), count] for (prof_id, rating, count) in ratings]
    ratings = {prof_id: (rating * count + 9.604 - 1.96 * math.sqrt(count * rating * (5 - rating) + 24.01)) /
                        (count + 3.8416) for (prof_id, rating, count) in ratings}

    all_ratings.update(ratings)
    # Group all section by course ID.
    professors_by_course = {key: list({section.professor_id for section in value}) for (key, value) in
                            itertools.groupby(sorted(all_sections.values(), key=lambda section: section.course_id),
                                              key=lambda section: section.course_id)}

    profs_list = [(course_id, prof_id) for course_id in professors_by_course
                  for prof_id in professors_by_course[course_id]]
    ratings_dict = {(course_id, prof_id): all_ratings[prof_id] / max_rating
                    if (max_rating := max(all_ratings[prof_id] for prof_id in professors_by_course[course_id])) > 0
                    else 1 for (course_id, prof_id) in profs_list}

    section_ratings = {section_id: ratings_dict[(section.course_id, section.professor_id)]
                       for (section_id, section) in all_sections.items()}


def standardize(arr: np.ndarray):
    """Normalize the array so that the maximum is assigned a value of 1 and the minimum a value of 0."""
    if len(arr) == 1:
        return np.ones(1)
    arrmax, arrmin = arr.max(initial=-math.inf), arr.min(initial=math.inf)
    return (arr - arrmin) / (arrmax - arrmin)


@app.before_request
def load_student():
    """Instantiate objects containing the available intervals and the student login info"""
    global available_intervals
    g.user = None
    student_id = session.get("student_id")
    if student_id:
        student = Student.query.get(student_id)
        g.user = student
        all_available_intervals = AvailableTimes.query.filter(AvailableTimes.student_id == g.user.id).all()
        days = ["monday", "tuesday", "wednesday", "thursday", "friday"]
        available_intervals = {days[key - 1]: list(chunks(list(
            flatten([[[v.start, 'S'], [v.end, 'E']] for v in value])), 2)) for (key, value) in itertools.groupby(
            sorted(all_available_intervals, key=lambda interval: (interval.day, interval.start)),
            key=lambda interval: interval.day)}
        # Now to convert that to start and end:
        # Dict of days, intervals


# The index route would provide the "basic" functionalities of the course selection.
@app.route('/', methods=['POST', 'GET'])
def index():
    """The main page to search for courses"""
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
            page_num = 0
        if 'page' in request.form:
            if 'previous' in request.form:
                page_num = max(0, page_num - 1)
            elif 'next' in request.form:
                page_num = min((len(results) - 1) // 25, page_num + 1)
            else:
                page_num = int(request.form['nav']) - 1

    selected = [all_courses[int(x)].title for x in courses]
    selected_abbreviations = [all_courses[int(x)].abbreviation for x in courses]
    return render_template('/index.html', courses=list(all_courses.values()), selected_courses=selected,
                           query=display(results, page_num), page=page_num, end=min(25 * (page_num + 1), len(results)),
                           num_pages=(len(results) + 24) // 25, results=results, user=g.user,
                           selected_abbreviations=selected_abbreviations, schedules=all_schedules,
                           professors=all_professors)


@app.route('/login.html', methods=['POST', 'GET'])
def login():
    """The login page"""
    error = None
    if request.method == 'POST':
        # Usual credentials (email and password)
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
    """The sign-up page designed to create accounts"""
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

        # Ensure there are no duplicates.
        existing_email = Student.query.filter(Student.email == email).first()
        if existing_email:
            error = "Email has already been taken."
        elif repeat_password != password:
            error = "Passwords must match."
        else:
            # If there are no errors, then the student will be added to the database.
            student = Student(email=email, password=hashed_password, first_name=first,
                              last_name=last, middle_initial=middle, major="Undecided")
            db.session.add(student)
            db.session.commit()
            session['student_id'] = student.id

            return redirect(url_for("profile"))
    return render_template('/signup.html', error=error)


# Dict, sorted from Monday to Friday, Start to End, with respect to X.
def get_intervals(preferences):
    """Creates a dict of preferences from the dynamically programmed inputs."""
    keys = list(preferences.keys())
    result = {'monday': [], 'tuesday': [], 'wednesday': [], 'thursday': [], 'friday': []}
    days = [key for key in result.keys()]
    day = 0
    # key: a string that has a day and a start/end trigger.
    for i in range(len(keys) // 2):
        if not keys[2 * i].startswith(days[day]):
            day += 1
        start = preferences[keys[2 * i]]
        end = preferences[keys[2 * i + 1]]
        result[days[day]].append([start, end])

    for d in days:
        # Sort by start time (the first time listed is the start time).
        result[d] = sorted(result[d], key=lambda x: get_time_from_str(x[0]))

    return result


def get_time_from_str(time_str):
    hours, minutes = time_str.split(':')[:2]
    return 60 * int(hours) + int(minutes)


def get_time_in_minutes(time):
    return 60 * time.hour + time.minute


@app.route('/profile', methods=['POST', 'GET'])
def profile():
    # Preferences
    error, message = None, None
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
            # The pairwise method lists all overlapping pairs such that a collection of items S maps to
            # (S[0], S[1]), (S[1], S[2]), and so on.
            for first, second in itertools.pairwise(intervals[key]):
                # An overlapping interval starts later than the earlier interval ends.
                if get_time_from_str(second[0]) < get_time_from_str(first[1]):
                    overlap = True
                    break
            else:
                continue
            break
        if not overlap:
            AvailableTimes.query.filter_by(student_id=g.user.id).delete()
            for idx, day in enumerate(intervals, start=1):
                for interval in intervals[day]:
                    preference = AvailableTimes(student_id=g.user.id, day=idx,
                                                start=interval[0], end=interval[1])
                    db.session.add(preference)
            db.session.commit()
            message, error = "Your preferences have been successfully saved.", None
        else:
            message, error = None, "Time intervals cannot overlap each other."

    # Group every interval by day
    days = ["monday", "tuesday", "wednesday", "thursday", "friday"]
    prefs = AvailableTimes.query.filter(AvailableTimes.student_id == g.user.id).all()
    pref_dict = itertools.groupby(sorted(prefs, key=lambda pref: (pref.day, pref.start)), key=lambda pref: pref.day)
    pref_dict = {days[key - 1]: list(value) for (key, value) in pref_dict}

    prefs = {"monday": [], "tuesday": [], "wednesday": [], "thursday": [], "friday": []}
    prefs.update(pref_dict)
    return render_template('/profile.html', user=g.user, prefs=prefs, error=error, message=message)


# The classes corresponding to the tables in the database are shown below.
# Notice that each field of the class is a column that contains a data type.
class Course(db.Model):
    """Data model for courses"""

    __tablename__ = 'Course'

    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(150), unique=True, nullable=False)
    abbreviation = db.Column(db.String(10), nullable=False)
    category = db.Column(db.String(10), nullable=False)


class Corequisite(db.Model):
    """Data model for co-requisite courses (to be taken with the course)."""

    __tablename__ = 'Corequisite'

    id = db.Column(db.Integer, primary_key=True)
    course_id = db.Column(db.Integer, db.ForeignKey('Course.id'))
    course = db.relationship('Course', foreign_keys=course_id)
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


class Section(db.Model):
    """Some info about the course section."""
    __tablename__ = 'Section'
    id = db.Column(db.Integer, primary_key=True)
    section = db.Column(db.String(5), nullable=False)
    course_id = db.Column(db.Integer, db.ForeignKey('Course.id'), nullable=False)
    course = db.relationship('Course', foreign_keys=course_id)
    professor_id = db.Column(db.Integer, db.ForeignKey('Professor.id'), nullable=False)
    professor = db.relationship('Professor', foreign_keys=professor_id)
    seats = db.Column(db.Integer, nullable=False)
    min_pass_grade = db.Column(db.String(2))


class Schedule(db.Model):
    """Some info about the coursework"""
    __tablename__ = 'Schedule'

    id = db.Column(db.Integer, primary_key=True)
    section_id = db.Column(db.Integer, db.ForeignKey('Section.id'), nullable=False)
    section = db.relationship('Section', foreign_keys=section_id)
    start = db.Column(db.Time, nullable=False)
    end = db.Column(db.Time, nullable=False)
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

# Code to handle influx of data from CSVs to store onto the database.
#
# with open('static/data/Professor.csv', 'r') as file:
#     for row in csv.DictReader(file):
#         db.session.add(Professor(last_name=row['Last'], first_name=row['First'], middle_initial=row['Middle']))
#
# with open('static/data/Review.csv', 'r') as file:
#     for row in csv.DictReader(file):
#         db.session.add(Review(student_id=int(row['Student ID']), professor_id=int(row['Professor ID']),
#                               content=row['Content'], rating=row['Rating']))
#
# with open('static/data/Courses.csv', 'r') as file:
#     for row in csv.DictReader(file, dialect='excel-tab'):
#         db.session.add(Course(title=row['Title'], category=row['Category'], abbreviation=row['Abbreviation']))
#
# db.session.commit()
#
# with open('static/data/Section.csv', 'r') as file:
#     for row in csv.DictReader(file):
#         db.session.add(Section(course_id=row['Course ID'], professor_id=row['Professor ID'], section=row['Section'],
#                                seats=int(row['Seats'])))
#
# db.session.commit()
#
# with open('static/data/Schedule.csv', 'r') as file:
#
#     def test(x):
#         return x.lower() == 'true'
#
#     for row in csv.DictReader(file):
#         db.session.add(Schedule(section_id=row['Section ID'], start=row['Start'], end=row['End'],
#                                 mondays=test(row['Mondays']), tuesdays=test(row['Tuesdays']),
#                                 wednesdays=test(row['Wednesdays'].lower()),
#                                 thursdays=test(row['Thursdays']), fridays=test(row['Fridays'])))
#
# db.session.commit()

all_courses = {course.id: course for course in Course.query.all()}
all_professors = {prof.id: prof for prof in Professor.query.all()}


# If you've made it this far into the code, congratulations.
def mutate(genome, indices, rules):
    """The mutation function will replace some course sections with new course sections if possible."""
    for i, j in enumerate(indices):
        # Remove the original from the list of rules.
        lst = list(set(rules[i]) - {genome[j]})

        # If there are other numbers then the genome will be modified.
        if len(lst):
            genome[j] = random.choice(lst)


def guest_algorithm(selected_courses):
    """
    The guest algorithm runs without taking any preferences into account. It minimizes the number of
    possible combinations of sections by testing for overlap and doesn't take much time to run.

    Steps
    ----

    1\\. Generate the list of possible combinations of proposed course schedules.

    2\\. If there are less than 50000 possibilities, only keep the non-overlapping schedules as courses are added.

    3\\. If there are more than 50000 possibilities, then randomly generate a possibility and check for overlap.
    This may return a different result for the same query.

    4\\. Return the first 1000 non-overlapping course schedules.
    """
    # Step 1
    sections = Section.query.filter(Section.course_id.in_(selected_courses)).order_by(Section.course_id.asc(),
                                                                                      Section.id.asc()).all()
    rules = [[section.id for section in course_sections] for _, course_sections
             in itertools.groupby(sections, key=lambda section: section.course_id)]
    result = set()
    genomes = [[]]
    product = np.prod([float(len(rule)) for rule in rules])

    # Step 2
    if product < 50000:
        for count, r in enumerate(rules):
            genomes = itertools.product(genomes, r)
            genomes = list(flatten(genomes))
            genomes = [genome for genome in chunks(genomes, count + 1) if overlap_test(genome)]

        result = random.sample(genomes, k=min(1000, len(genomes)))
        # Step 4
        return [([all_sections[section_id] for section_id in genome], 100) for genome in result]

    # Step 3
    trials = 0
    while len(result) < 1000 and trials < 50000:
        genome = [random.choice(rule) for rule in rules]
        if overlap_test(genome):
            result.add((*genome,))
        trials += 1

    # Step 4
    return [(Section.query.filter(Section.id.in_(genome)).order_by(Section.id.asc()).all(), 100) for genome in
            result]


def display(query, page=0):
    return query[(25 * page):(25 * (page + 1))]


def flatten(lst):
    """Takes a list and 'flattens' it so that there are no nested lists."""
    for i in lst:
        yield from [i] if not isinstance(i, tuple) and not isinstance(i, list) else flatten(i)


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


# Genome: A list of numbers that concern the course section IDs.
# Gene: The ID of that specific course section
def brute_force(lst, alpha):
    """The brute-force approach is the most straightforward.
    It is used when there is a sufficiently small number
    of combinations of courses (order less than ~250k).

    Steps
    ----
    1\\. Filter all sections by course ID so that only the relevant sections remain.

    2\\. Create a population P of all possible combinations (genomes) of the course sections.

    3\\. Calculate the fitness function of all genomes of P and sort all members by their fitness scores.

    4\\. Filter every genome whose score is greater than zero.

    5\\. Take the top solutions whose scores are within 5% of the maximum score.
    """
    global all_ratings

    # Step 1
    lst_courses = [[x for (x,) in Section.query.with_entities(Section.id).filter(
        Section.course_id == course).all()] for course in lst]

    # Step 2
    population = np.array(list(map(list, itertools.product(*lst_courses))))

    # Step 3
    fitness_func = np.vectorize(fitness, signature='(n),() -> ()')
    # start = timeit.default_timer()
    scores = fitness_func(population, alpha)
    # print(timeit.default_timer() - start)

    # Step 4
    possibilities = [(genome, score) for (genome, score) in zip(map(tuple, population), scores) if score > 0]
    possibilities = sorted(possibilities, key=lambda x: x[1], reverse=True)
    max_score = max(scores, default=0)

    # Step 5
    result = dict(itertools.takewhile(lambda x: x[1] >= 0.95 * max_score, possibilities))
    sections = [(all_sections[section_id] for section_id in sorted(genome)) for genome in result.keys()]
    return [(tuple(genome), f"{score:0.1f}") for genome, score in zip(sections, result.values())]


def crossover(genome1, genome2):
    """Crosses the two genomes by swapping part of the genome with the other at a random point.
    Genome lengths must be equal."""
    assert len(genome1) == len(genome2), "Lengths are not equal."
    if len(genome1) <= 1:
        return genome1, genome2
    point = random.randint(1, len(genome1) - 1)
    return genome1[:point] + genome2[point:], genome2[:point] + genome1[point:]


def overlap_test(genome):
    """
    1\\. The schedules are extracted from each section and the dict of intervals is created.

    2\\. The schedules are then placed onto the dict of intervals, noting the day of the interval and whether
    they represent the start or end of an interval, sorted by the start time in each bin.

    3\\. Check whether there is an overlap in the course time by iterating through each "bin" in the dict. Schedules
    with at least one overlap will return False.


    :param genome: The sequence of course sections to test for overlap.
    :return: True only if the no course times overlap.
    """
    # Step 1
    intervals = {'monday': [], 'tuesday': [], 'wednesday': [], 'thursday': [], 'friday': []}
    days = [key for key in intervals.keys()]

    global all_schedules, all_sections

    # Step 2
    sections = [all_sections[section_id] for section_id in genome]
    schedules = list(flatten([all_schedules[section.id] for section in sections]))
    if not len(sections):
        raise ValueError('No course section found. Please try again.')
    for schedule in schedules:
        scheduled_days = [schedule.mondays, schedule.tuesdays, schedule.wednesdays,
                          schedule.thursdays, schedule.fridays]
        scheduled_times = [schedule.start, schedule.end]
        for day in [day for day, scheduled in zip(days, scheduled_days) if scheduled]:
            intervals[day].append(scheduled_times)

    intervals = {day: sorted(intervals[day], key=lambda x: x[0]) for day in intervals}

    # Step 3
    for key in intervals:
        for i in range(len(intervals[key]) - 1):
            # An overlapping interval starts later than the earlier interval ends.
            if intervals[key][i + 1][0] < intervals[key][i][1]:
                return False
    return True


def diff_time(end, start):
    return (end.hour - start.hour) * 60 + (end.minute - start.minute)


def fitness(genome: np.ndarray, alpha: float):
    """
    :param alpha: A parameter that weighs the priorities of a preferred schedule to a highly rated professor.

    :param genome: A sequence of section IDs.

    Calculates the fitness score from 0 to 100 of a schedule based on the user preferences.
    The steps of calculating the fitness score are as follows:

    1\\. The schedules are extracted from each section and the dict of intervals is created.

    2\\. The schedules are then placed onto a dict of intervals, noting the day of the interval and whether
    they represent the start or end of an interval, sorted by the start time in each bin.

    3\\. Check whether there is an overlap in the course time by iterating through each "bin" in the dict. Schedules
    with at least one overlap will be heavily penalized (i.e. return a value of 0).

    4\\. "Merge" the availability times with the dict of intervals and calculate the ratio of overlap time to the total
    course time. The ratio will be multiplied by 100 to obtain the flexibility score. (Skip this step and assign
    0 to the flexibility score if alpha = 0)

    5\\. Compare the ratings of each professor in each section by dividing the corresponding adjusted rating by the max
    rating of all professors teaching the same course. The rating score is calculated by the mean of each adjusted
    rating and multiplying it by 100. (Skip this step and assign 0 to the rating score if alpha = 1)

    6\\. Return alpha * (flexibility score) + (1 - alpha) * (rating score) only if there is no overlap between the
    course intervals.

    Special Cases
    ----


    alpha = 1: Only factor in schedule flexibility.

    alpha = 0: Only factor in professor ratings.
    """
    # Step 1
    global all_schedules, available_intervals, schedules_by_section
    intervals = {'monday': [], 'tuesday': [], 'wednesday': [], 'thursday': [], 'friday': []}

    days = [key for key in intervals.keys()]
    schedules = list(flatten(map(schedules_by_section.get, map(all_sections.get, genome))))
    # Equivalently: schedules = [[all_schedules[section.id] for section in all_sections]]

    # Step 2
    course_time = 0
    for schedule in schedules:
        scheduled_days = [schedule.mondays, schedule.tuesdays, schedule.wednesdays,
                          schedule.thursdays, schedule.fridays]
        start, end = schedule.start, schedule.end
        course_time += (60 * (end.hour - start.hour) + (end.minute - start.minute)) * sum(scheduled_days)
        for idx, day in enumerate(days):
            if scheduled_days[idx]:
                intervals[day].extend([[start, 'S'], [end, 'E']])
    if not len(schedules):
        raise TypeError('No course section found. Please try again.')
    intervals = {day: sorted(intervals[day], key=lambda x: x[0]) for day in intervals}

    # Step 3
    for day_intervals in intervals.values():
        count = 0
        for interval in day_intervals:
            # An overlapping interval starts later than the earlier interval ends.
            count += 1 if interval[1] == "S" else -1
            if count > 1:
                return 0
    # Step 4
    if alpha == 0:
        return 100 * max(sum(section_ratings[section_id] for section_id in genome) / len(genome), 1)

    for key, value in available_intervals.items():
        intervals[key] += value
    intervals = {day: sorted(intervals[day], key=lambda x: x[0]) for day in intervals}
    overlap = 0
    for value in intervals.values():
        count = 1
        for (first, second) in itertools.pairwise(value):
            # An overlapping interval starts later than the earlier interval ends.
            if count > 1:
                overlap += diff_time(second[0], first[0])
            count += 1 if second[1] == "S" else -1
    # Steps 5 & 6
    if alpha == 1:
        return 100 * overlap / total_time if (total_time := max(overlap, course_time)) > 0 else 100
    return 100 * ((alpha * (overlap / total_time) if (total_time := max(overlap, course_time)) > 0 else 1) +
                  ((1 - alpha) * sum(section_ratings[section_id] for section_id in genome) / max(len(genome), 1)))


def genetic_algorithm(selected_courses, alpha):
    """
    :param selected_courses: The user-provided input of selected courses.
    :param alpha: A parameter that weighs the preferences of a better professor against a more flexible timeline.
    :return: A list of schedules of courses with their scores.

    The genetic algorithm will return the brute force if it is slower than the brute force algorithm.

    Alpha is a parameter that takes into account the preferences of a better professor
    vs. a timeline more flexible. The courses are based on the user's preferences.
    If the ratings don't account in the decision to take some course,
    the alpha value should be set to 1.0. Of course, if only ratings
    matter then the alpha value should be set to 0. By default, alpha is 0.8,
    which means that the program will prioritize availability constraints over
    better professor ratings.

    When the number of possibilities is ~250k, the speed of the genetic algorithm is
    comparable to that of the brute force algorithm.

    Steps
    ----
    1\\. Initialize rules and calculate the number of combinations of courses.

    2\\. The genetic algorithm will branch to the brute force algorithm if there
    used when there are less than 200000 possibilities.

    3\\. Initialize the population, generating 250 random course schedule proposals.

    4\\. Vectorize the fitness function to add functionality to the numpy array.

    5\\. Apply the fitness function to the population and evaluate all genomes. Skip to Step 10
    if the fitness function of a genome returns a value close enough to 100.

    6\\. Standardize the fitness scores (score - min) / (max - min) and add 0.1 to the resultant score. This number
    will be the corresponding weight of the genome. Thus, a higher fitness score will be favored over a lower fitness
    score.

    7\\. Crossover the parents to make new genomes and mutate them. The mutation process is important to finding
    different solutions. The number of individuals should be 250 for the next generation. Due to the uniqueness
    of the population, however, it should be noted that if there are less than 50 individuals, then there must be more
    offspring to compensate for the lack of diversity.

    8\\. The new population will consist of the crossover mutations, and the top 50 fitness scores are guaranteed a
    spot in the next generation.

    9\\. Steps 5-8 will be repeatedly applied until (a) 500 generations have elapsed or (b) there is a genome with a
    high enough fitness score.

    10\\. Select all genomes whose score is within 5% of the maximum fitness score. There is a chance
    that a lower match rate will give a better schedule than a higher match rate as long
    as not much is being sacrificed.

    """

    # Step 1
    result, rules = [], [[int(x) for (x,) in Section.query.with_entities(Section.id)
                         .filter(Section.course_id == course).all()] for course in selected_courses]
    # Equivalently: possibilities = np.prod([len(rule) for rule in rules])
    possibilities = np.prod(list(map(len, rules)))

    # Step 2
    if possibilities < 250000:
        return brute_force(selected_courses, alpha)

    # Step 3
    global all_ratings
    population = np.array([[random.choice(rule) for rule in rules] for _ in range(250)])

    # Step 4
    fitness_func = np.vectorize(fitness, signature='(n),() -> ()')

    # Used for benchmarking purposes.
    # start_time = timeit.default_timer()

    for i in range(500):
        # Step 5
        scores = fitness_func(population, alpha)

        scoreboard = {tuple(k): v for (k, v) in sorted(zip(population, scores), key=lambda x: x[1], reverse=True)}
        population, scores = list(scoreboard.keys()), list(scoreboard.values())

        # Used for benchmarking purposes.
        # if i % 100 == 99:
        #     print(f'{timeit.default_timer() - start_time}s to evaluate 100 generations')
        #     print(f'Iteration {i + 1}, {scoreboard[next(iter(scoreboard))]}')
        #     start_time = timeit.default_timer()

        #
        if scoreboard[next(iter(scoreboard))] >= min(99, 100 - i // 100):
            break

        # Step 6
        chances = standardize(np.array(scores)) + 0.1

        # Step 7
        fathers = random.choices(population=population, weights=chances, k=max(100, 125 - len(population) // 2))
        mothers = random.choices(population=population, weights=chances, k=max(100, 125 - len(population) // 2))
        offspring = [crossover(mother, father) for father, mother in zip(fathers, mothers)]
        offspring = [list(x) for xs in offspring for x in xs]  # Unpack a list of lists

        for genome in offspring:
            # There is a 1% chance that a specific gene will be mutated.
            num_genes_mutated = np.random.binomial(7, 0.01)
            genes_mutated = random.sample(range(len(selected_courses)), k=num_genes_mutated)
            if num_genes_mutated:
                mutate(genome, genes_mutated, [rules[gene] for gene in genes_mutated])

        # Step 8
        population = [x for x in list(scoreboard)[:50] + offspring]
        if len(population) > 100:
            population.pop()
    # Step 10
    scores = fitness_func(population, alpha)
    scoreboard = sorted(list(zip(map(tuple, population), scores)), key=lambda x: x[1], reverse=True)
    max_score = scoreboard[0][1]
    result = dict(itertools.takewhile(lambda x: x[1] >= 0.95 * max_score, scoreboard))
    return [(genome, f"{score:0.1f}") for genome, score in zip([Section.query.filter(Section.id.in_(genome))
                                                               .order_by(Section.id.asc()).all()
                                                                for genome in result.keys()], list(result.values()))]


all_ratings, all_sections, all_schedules, professors_by_course, available_intervals = {}, {}, [], [], {}
section_ratings, schedules_by_section = {}, {}


if __name__ == '__main__':
    app.run()
