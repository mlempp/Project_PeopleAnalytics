# Project1

#IBM HR Analytics Employee Attrition & Performance

In this project I wanted to predict attrition based on employee data.
The data is an artificial dataset from IBM data scientists. It contains data for 1470 employees. Te dataet contains the following information per emplyee:

Age                         : Age of the employee.

Attrition                   : If the employee left or not. 
BusinessTravel              : If the employee has to travel. 
DailyRate                   : The daily rate of the employee.
Department                  : The department where the employee is working in.
DistanceFromHome            : How far the emplyee has to travel per day.
Education                   : The highest degree the emplyee reached. (1 'Below College', 2 'College', 3 'Bachelor', 4 'Master', 5 'Doctor')
EducationField              : In which field the employee graduated.
EmployeeCount               : Counter (all 1)
EmployeeNumber              : Individual identifier of the employee.
EnvironmentSatisfaction     : The employee satisfaction with the environemnt. (1 'Low', 2 'Medium', 3 'High', 4 'Very High')
Gender                      : The geneder of the employee.
HourlyRate                  : The hourly rate of the employee.
JobInvolvement              : The level oft the involvement in the job. (1 'Low', 2 'Medium', 3 'High', 4 'Very High')
JobLevel                    : The job level.
JobRole                     : The role of the employee.
JobSatisfaction             : The employee satisfaction with the job. (1 'Low', 2 'Medium', 3 'High', 4 'Very High')
MaritalStatus               : If the employee is married or not.
MonthlyIncome               : The monthly salary of the employee.
MonthlyRate                 : The monthly rate of the employee.
NumCompaniesWorked          : In how many companies the employee worked before.
Over18                      : If the employee is over 18 yeary old.
OverTime                    : If the employee does overhours.
PercentSalaryHike           : The percentage of the last raise.
PerformanceRating           : The performance rating of the employee. (1 'Low', 2 'Good', 3 'Excellent', 4 'Outstanding')
RelationshipSatisfaction    : The employee satisfaction with the private relationship. (1 'Low', 2 'Medium', 3 'High', 4 'Very High')
StandardHours               : Number of hours the employee has to work.
StockOptionLevel            : If the employee has stock of the company.
TotalWorkingYears           : How long the employee is working already in total.
TrainingTimesLastYear       : How many trainings the employee had last year.
WorkLifeBalance             : How the employee rates the work-life-balance. (1 'Bad', 2 'Good', 3 'Better', 4 'Best')
YearsAtCompany              : How long the employee works for the current company.
YearsInCurrentRole          : How long the employee works in the current role.
YearsSinceLastPromotion     : How long since the last promotion of the employee.
YearsWithCurrManager        : How long the employee works with the current manager.

The provider of the data does not give any more information on the features.



#EDA#

I start with an simple exploratory data anylsis.

1. Check the datatypes ad if we have missing data in the dataset.
2. Transform the attrition to a numerical value (1 Yes, 0 No) and drop some features (Over18, EmployeeCount, StandardHours).
3. Make exploatory plots.

Summary:
- Turnover is 16.12 %, and with that relatively high.
- The longer in the company/older the the unlikely the leave.
- The following features lead to higher attrition:
	BusinessTravel              Travel_Frequently
	Department                  Sales
	EducationField              Technical Degreee, Human Resource, Marketing
	Gender                      Male
	JobRole                     Laboratory Technician, Sales Representataive, Human Rersources
	MaritalStatus               Single
	OverTime                    Yes

4. Transform the categories to numericals
5. Check the correlation across the different features.

Summary: 
	- Monthly income correlates with job level.
	- PerformanceRating correlates with PercentSalaryHike.
	- Department correlates with role.

6. Check if the coocurance of different features at different levels influences the attrition
	a) Age + MonthlyIncome
	b) OverTime, MonthlyIncome
	c) WorkLifeBalance, RelationshipSatisfaction
	d) BusinessTravel, DistanceFromHome




#prepare the data for the model#

1. Split the data in train and test (80-20)
2. Oversample the train data




#train and optimize 3 different gradient-boosted tree based models#

1. RandomForestClassifier
2. LightGBM
3. XGboost
4. Bagging the estimations of te three models

Summary:
RandomForest:	AUC:86.394558 F1:85.534844
LightGBM:	AUC:85.714286 F1:85.584242
XGboost:	AUC:85.714286 F1:84.628313

Bagging: 	AUC:87.074830 F1:86.092283

5. Plot the metrics



#inspect the most important features for the leaving the company#

1. Plot the feature importance

