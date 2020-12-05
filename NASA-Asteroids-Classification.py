# Importing Libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Reading the Data
dataset = pd.read_csv('./Data/nasa.csv', sep=',', na_values=["n/a", "na", "--"])

# Cheking Missing Values
dataset.isnull().any()

X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

# Encoding Object columns in the right format
X['Close Approach Date'] = pd.to_datetime(X['Close Approach Date'])
X['Orbiting Body'] = LabelEncoder().fit_transform(X['Orbiting Body'])
X['Orbit Determination Date'] = pd.to_datetime(X['Orbit Determination Date'])
X['Equinox'] = LabelEncoder().fit_transform(X['Orbiting Body'])
y = LabelEncoder().fit_transform(y)

X.info()

# Droping 0 Variance Columns
variance = X.var()

for col in X.var().index:
    if variance[col] == 0:
        print(col)
        X.drop(col, axis=1, inplace=True)
        
# Dropping useless features
    ######## Here are the Features Description ########
    # Neo Reference ID - Near Earth Object (NEO) reference ID number for an asteroid (or a comet) which comes close to earth
    # Name - 'Name' of asteroid (same as NEO Reference ID)
    # Absolute Magnitude - A measure of the asteroid's luminosity (in H) (the brightness of an asteroid if it is 1 astronomical unit away from both the Sun and the observer, and the angle between the Sun, asteroid, and Earth is 0 degrees)
    # Est Dia in (in KM, M, Miles, and Feet) (min) - Minimum estimated diameter of the asteroid (Note: Since asteroids cannot be directly measured and because they have irregular shapes, their diameters are estimates. These estimates are calculated using its absolute magnitude and geometric albedo.)
    # Est Dia in (in KM, M, Miles, and Feet) (max) - Maximum estimated diameter of the asteroid
    # Close Approach Date - Date at which the asteroid approaches close to Earth
    # Epoch Date Close Approach - Date at which the asteroid approaches close to Earth (in epoch time)
    #   Relative Velocity (in km per sec, km per hr, and miles per hour) - Asteroid's velocity relative to earth
    #   Miss Dist.(in Astronomical, lunar, km, and miles) - Distance by which the asteroid misses Earth
    # Orbiting Body - 
    # Orbit ID - An ID of JPL NEA orbit that JPL Nasa uses in its analysis
    # Orbit Determination Date - Date at which the asteroid's orbit was determined
        # Orbit Uncertainity - A measure of the uncertainity ('measurement errors') in the calculated orbit
    # Minimum Orbit Intersection - The closest distance between Earth and the asteroid in their respective orbits (in astronomical units)
    # Jupiter Tisserand Invariant - A value used to differentiate between asteroids and Jupiter-family comets
    # Epoch Osculation - The instance of time at which the asteroid's position and velocity vectors (from which its osculating orbit is calculated) is specified
    # Eccentricity - A value which specifies by how much the asteroid's orbit deviates from a perfect circle
    # Semi Major Axis - The longest radius of an elliptical orbit; a easure of the asteroid's average distance from the Sun (asteroids orbit the Sun)
    # Inclination - Measures the tilt of the asteroid's orbit around the Sun
    # Asc Node Longitude - (copying from NASA) 'Angle in the ecliptic plane between the inertial-frame x-axis and the line through the ascending node'
    # Orbital Period - Time taken for asteroid to complete a single orbit around the Sun
    # Perihelion Distance - Distance of point in asteroid's orbit which is closest to the Sun
    # Perihelion Arg - (copying from Nasa) 'The angle (in the body's orbit plane) between the ascending node line and perihelion measured in the direction of the body's orbit'
    # Aphelion Dist - Distance of point in asteroid's orbit which is farthest from the Sun
    # Perihelion Time - Length of time of asteroid's passage through the perihelion stage
    # Mean Anomaly - (copying from Nasa) 'The product of an orbiting body's mean motion and time past perihelion passage'
    # Mean Motion - (copying from Nasa) 'The angular speed required for a body to make one orbit around an ideal ellipse with a specific semi-major axis'
    # Equinox - An astronomical standard to measure against (currently 'J2000.0')
    # Hazardous - Is the asteroid hazardous? (True or False)
    #################################################################################### 

# - Drop Neo Reference ID and Name as they are just identifiers
X.drop(['Neo Reference ID', 'Name'], axis=1, inplace=True)

# - We have also the estimated diameter of the asteroid (max and min) in km, m, miles
#   and feets, we need just one of them so we will keep only the one with meters
X.drop(['Est Dia in KM(min)', 'Est Dia in KM(max)', 'Est Dia in Miles(min)', 
        'Est Dia in Miles(max)', 'Est Dia in Feet(min)', 'Est Dia in Feet(max)'], 
        axis=1, inplace=True)

# - we have Relative Velocity in km per sec, km per hr, and miles per hour
# we'll keep km per second only
X.drop(['Relative Velocity km per hr', 'Miles per hour'], axis=1, inplace=True)

# - we have also Miss Dist (Distance by which the asteroid misses Earth) in 
#   Astronomical, lunar, km, and miles, we'll keep km only
X.drop(['Miss Dist.(Astronomical)', 'Miss Dist.(lunar)', 
        'Miss Dist.(miles)'], axis=1, inplace=True)

# - we'll also drop Orbit Determination Date as there is no logical reason why
#   it should affect the target
X.drop('Orbit Determination Date', axis=1, inplace=True)

# - since Jupiter Tisserand Invariant is exactly calculate from semi-major axis, 
#   eccentricity and inclination (https://en.wikipedia.org/wiki/Tisserand%27s_parameter)
#   so there is no need to keep it because those three variables explain it perfectly 
X.drop('Jupiter Tisserand Invariant', axis=1, inplace=True)

# - Epoch Osculation is the instance of time at which the asteroid's position and 
#   velocity vectors is specified so we will drop it since it's not relevent 
#   to the target
X.drop('Epoch Osculation', axis=1, inplace=True)

# - Close Approach Date and Epoch Date Close Approach are the same measures in two 
#   different units, so keeping one of them is fine but i did a little bit of 
#   searching and i foundout that these two features do not impact the target 
#   so we will remove them 
X.drop(['Close Approach Date', 'Epoch Date Close Approach'], axis=1, inplace=True)






















