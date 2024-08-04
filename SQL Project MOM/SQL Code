create schema MKCL_MOM2023;
use MKCL_MOM2023;

-- Create a tables to store Learnes Information
CREATE TABLE LearnerNames (
    Id INT PRIMARY KEY AUTO_INCREMENT,
    Title VARCHAR(10),
    FirstName VARCHAR(50) NOT NULL,
    MiddleName VARCHAR(50),
    LastName VARCHAR(50) NOT NULL,
    NameOnCertificate VARCHAR(100)
    );
    
CREATE TABLE LearnerInfo (
Id INT PRIMARY KEY AUTO_INCREMENT,
Dob DATE,
Gender VARCHAR(10),
MaritalStatus VARCHAR(20),
ProofOfIdentity VARCHAR(50),
MotherTongue VARCHAR(50),
Qualification VARCHAR(50),
FOREIGN KEY (Id) REFERENCES LearnerNames (Id)
);
    
CREATE TABLE LearnerContactandAddress (
Id INT,
MobileSelf VARCHAR(50),
Address VARCHAR(50),
City VARCHAR(50),
Pin VARCHAR(50),
State VARCHAR(50),
District VARCHAR(50),
Taluka VARCHAR(50),
FOREIGN KEY (Id) REFERENCES LearnerNames (Id)
);

    
INSERT INTO LearnerInfo (id,Title,FirstName,MiddleName,LastName,NameOnCertificate,Dob,Gender,MaritalStatus,
						  ProofOfIdentity,MotherTongue,Address,City,Pin,MobileSelf,State,District,Taluka,
                          Qualification)
VALUES ('57','Mr','Aditya','Laxmikant','Pathak','Pathak Aditya Laxmikant','1999/07/21','Male','Single','Aadhar Card',
	     'Marathi','Dharashiv','Dharashiv','413501','8624800262','Maharashtra','Dharashiv','Dharashiv','MSc')


-- Show variables like "secure_file_priv";

-- Upload the content from the csv file into Corresponding Tables

Load Data Infile 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/learner.csv'
into table LearnerNames
fields terminated by ','
enclosed by '"'
lines terminated by '\n'
ignore 1 rows;

Load Data Infile 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/learnerinfo.csv'
into table LearnerInfo
fields terminated by ','
enclosed by '"'
lines terminated by '\n'
ignore 1 rows;

Load Data Infile 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/learneraddress.csv'
into table LearnerContactandAddress
fields terminated by ','
enclosed by '"'
lines terminated by '\n'
ignore 1 rows;

-- Create a table to store Learner Code an Marks
CREATE TABLE LearnerMarks (
Id INT PRIMARY KEY AUTO_INCREMENT,
course_name VARCHAR(100),
learner_code VARCHAR(50),
verification_code VARCHAR(10),
MarksSored VARCHAR(10),
FOREIGN KEY (Id) REFERENCES LearnerNames (Id)
);

Load Data Infile 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/learnermarks.csv'
into table LearnerMarks
fields terminated by ','
enclosed by '"'
lines terminated by '\n'
ignore 1 rows;

-- View Tables

SELECT * from LearnerNames;
SELECT * from LearnerInfo;
SELECT * from LearnerContactandAddress;
SELECT * from LearnerMarks;

-- To Drop Tables Perform Following Queries
DROP TABLE LearnerInfo;
DROP TABLE LearnerMarks;

-- To delete first row perform following Query
DELETE FROM LearnerInfo
ORDER BY id
LIMIT 1;


Describe LearnerInfo; -- will describe table view or stored procedure
-- It provides information about the structure of the object,
-- Including column names, data types, constraints, and other properties.

-- FOR JOINING TABLES
SELECT *
FROM LearnerNames
JOIN LearnerMarks ON LearnerNames.id = LearnerMarks.id;

SELECT *
FROM LearnerNames
JOIN LearnerInfo ON LearnerNames.Id = LearnerInfo.Id
JOIN LearnerContactandAddress ON LearnerInfo.Id = LearnerContactandAddress.Id
JOIN LearnerMarks ON LearnerContactandAddress.Id = LearnerMarks.Id;
