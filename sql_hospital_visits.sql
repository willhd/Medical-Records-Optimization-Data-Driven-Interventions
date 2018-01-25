sql_bloom_project



# all of the documents with dates where date lines up with patient_id and document id is not null 22314
# note some of these have null entries for notes
SELECT
   c.patient_id 
  ,d.text_reference
  ,c."dateTime" as date
FROM
  documents d 
JOIN 
    clinical_encounters c
ON
    c.document_id = d.id
WHERE 
  c.document_id is not null;

# not hospitalized or ER 8083 
SELECT
     c.patient_id 
    ,d.text_reference
    ,c."dateTime" as date
FROM
     documents d 
JOIN 
    clinical_encounters c
ON
    c.document_id = d.id
WHERE 
    c.document_id is not null
AND 
    c.notes NOT like ('')
AND
    c.patient_id NOT IN(  
SELECT DISTINCT patient_id 
    FROM clinical_encounters 
    WHERE notes NOT like ('') AND document_id IS NOT NULL AND patient_id IN (
          SELECT DISTINCT patient_id 
          FROM  clinical_encounters 
          WHERE (lower(notes) like ('%emergency%')) 
          or 
          (lower(notes) like lower(('%recent hosp%'))) 
          or 
          (lower(notes) like lower(('% ER%'))) 
          or
          (lower(notes) like lower(('%discharged%'))) 
          or 
          (lower(notes) like lower(('%discharge summary%'))) 
          or
          (lower(notes) like lower(('%inpatient%'))) 
          or 
          (lower(notes) like lower(('%hospital f%'))) 
          or
          (lower(notes) like lower(('%hosp f%')))
          ));

# hospitalized patinet population 1297
SELECT
   c.patient_id 
  ,d.text_reference
  ,c."dateTime" as date
FROM
  documents d 
JOIN 
    clinical_encounters c
ON
    c.document_id = d.id
WHERE 
  c.document_id is not null
AND 
	c.notes NOT like ('')
    c.patient_id IN(  
SELECT DISTINCT patient_id 
    FROM clinical_encounters 
    WHERE notes NOT like ('') AND document_id IS NOT NULL AND patient_id IN (
          SELECT DISTINCT patient_id 
          FROM  clinical_encounters 
          WHERE (lower(notes) like ('%emergency%')) 
          or 
          (lower(notes) like lower(('%recent hosp%'))) 
          or 
          (lower(notes) like lower(('% ER%'))) 
          or
          (lower(notes) like lower(('%discharged%'))) 
          or 
          (lower(notes) like lower(('%discharge summary%'))) 
          or
          (lower(notes) like lower(('%inpatient%'))) 
          or 
          (lower(notes) like lower(('%hospital f%'))) 
          or
          (lower(notes) like lower(('%hosp f%')))
          ));

# dates of encounters joined with documents
SELECT
  c.patient_id 
  ,d.text_reference
  ,c."dateTime" as date
FROM
  documents d
LEFT JOIN 
    clinical_encounters c
ON
    d.emr_instance_id = c.emr_instance_id
WHERE
  d.text_reference IS NOT NULL limit 100


# patients who have not been to hospital or ER ordered by date: 8836
SELECT DISTINCT 
    patient_id, notes
    , "dateTime"  
    FROM 
        clinical_encounters 
    WHERE
         notes NOT like ('') 
         AND 
         patient_id 
         NOT IN (
            SELECT DISTINCT patient_id 
            FROM  clinical_encounters 
            WHERE 
            (lower(notes) like ('%emergency%')) 
            or 
            (lower(notes) like lower(('%recent hosp%'))) 
            or 
            (lower(notes) like lower(('%ER %'))) 
            or 
            (lower(notes) like lower(('% ER%'))) 
            or
            (lower(notes) like lower(('% ED%'))) 
            or
            (lower(notes) like lower(('%discharged%'))) 
            or 
            (lower(notes) like lower(('%inpatient%'))) 
            or 
            (lower(notes) like lower(('%pre-op%'))) 
            or
            (lower(notes) like lower(('%post-op%'))) 
            or
            (lower(notes) like lower(('%hospital f%'))) 
            or
            (lower(notes) like lower(('%hosp f%')))
            )
         ORDER BY "dateTime" DESC;

# selection as above with no null value for document reference: 7890
SELECT DISTINCT patient_id, notes, "dateTime", document_id  
    FROM clinical_encounters 
    WHERE notes NOT like ('') AND document_id IS NOT NULL AND patient_id NOT IN (
          SELECT DISTINCT patient_id 
          FROM  clinical_encounters 
          WHERE (lower(notes) like ('%emergency%')) or 
          (lower(notes) like lower(('%recent hosp%'))) or 
          (lower(notes) like lower(('%ER f%'))) or 
          (lower(notes) like lower(('% ER%'))) or
          (lower(notes) like lower(('% E/R%'))) or
          (lower(notes) like lower(('%discharged%'))) or 
          (lower(notes) like lower(('%inpatient%'))) or 
          (lower(notes) like lower(('%pre-op%'))) or
          (lower(notes) like lower(('%post-op%'))) or
          (lower(notes) like lower(('%hospital f%'))) or
          (lower(notes) like lower(('%hosp f%'))) 
          )ORDER BY "dateTime" ;

# Deep search for all possible ER/ Hosp related records 17450
SELECT DISTINCT patient_id, notes, "dateTime"  
    FROM clinical_encounters 
    WHERE notes NOT like ('') AND patient_id IN (
          SELECT DISTINCT patient_id 
          FROM  clinical_encounters 
          WHERE (lower(notes) like ('%emergency%')) or 
          (lower(notes) like lower(('%recent hosp%'))) or 
          (lower(notes) like lower(('%ER %'))) or 
          (lower(notes) like lower(('% ER%'))) or
          (lower(notes) like lower(('%discharged%'))) or 
          (lower(notes) like lower(('%inpatient%'))) or 
          (lower(notes) like lower(('%hospital f%'))) or
          (lower(notes) like lower(('%hosp f%')))
          )ORDER BY "dateTime" DESC;

# as above with no null values for  document_id: 1297
SELECT DISTINCT patient_id, notes, "dateTime", document_id  
    FROM clinical_encounters 
    WHERE notes NOT like ('') AND document_id IS NOT NULL AND patient_id IN (
          SELECT DISTINCT patient_id 
          FROM  clinical_encounters 
          WHERE (lower(notes) like ('%emergency%')) or 
          (lower(notes) like lower(('%recent hosp%'))) or 
          (lower(notes) like lower(('% ER%'))) or
          (lower(notes) like lower(('%discharged%'))) or 
          (lower(notes) like lower(('%discharge summary%'))) or
          (lower(notes) like lower(('%inpatient%'))) or 
          (lower(notes) like lower(('%hospital f%'))) or
          (lower(notes) like lower(('%hosp f%')))
          )ORDER BY "dateTime";

 #more precise version of above for hospitalized patient population: 133
 SELECT DISTINCT 
 	patient_id
 	,notes
 	,"dateTime"
 	,document_id  
    FROM 
    clinical_encounters 
    WHERE 
    notes NOT like ('') 
    AND 
    document_id IS NOT NULL 
    AND 
    patient_id 
    IN (
    SELECT DISTINCT 
    patient_id
    FROM  
    clinical_encounters 
    WHERE 
    (lower(notes) like ('%emergency depart%')) 
    or 
    (lower(notes) like lower(('%recent hospitalization%'))) 
    or 
    (lower(notes) like lower(('% ER f%'))) 
    or
    (lower(notes) like lower(('%discharge summary%'))) 
    or  
    (lower(notes) like lower(('%hospital follow%')))
     or
    (lower(notes) like lower(('%hosp follow%')))
    )
    ORDER BY "dateTime";

# problems associated with each hospitalized patient 
SELECT * 
  FROM clinical_problems
  where patient_id IN (
    SELECT DISTINCT patient_id
    FROM  clinical_encounters 
    WHERE (lower(notes) like ('%emergency depart%')) or 
    (lower(notes) like lower(('%recent hospitalization%'))) or 
    (lower(notes) like lower(('% ER f%'))) or
    (lower(notes) like lower(('%discharge summary%'))) or  
    (lower(notes) like lower(('%hospital follow%'))) or
    (lower(notes) like lower(('%hosp follow%'))));

# problemsfor each patient that started prior to thier hosp/ er visit 
SELECT *
FROM 
(SELECT DISTINCT clinical_encounters.patient_id, clinical_encounters.notes, clinical_encounters."dateTime"  
    FROM clinical_encounters
    WHERE notes NOT like ('') AND patient_id NOT IN (
          SELECT DISTINCT patient_id 
          FROM  clinical_encounters 
          WHERE (lower(notes) like ('%emergency%')) or 
          (lower(notes) like lower(('%recent hosp%'))) or 
          (lower(notes) like lower(('%ER %'))) or 
          (lower(notes) like lower(('% ER%'))) or
          (lower(notes) like lower(('% ED%'))) or
          (lower(notes) like lower(('%discharged%'))) or 
          (lower(notes) like lower(('%inpatient%'))) or 
          (lower(notes) like lower(('%pre-op%'))) or
          (lower(notes) like lower(('%post-op%'))) or
          (lower(notes) like lower(('%hospital f%'))) or
          (lower(notes) like lower(('%hosp f%')))
          )) 
          AS encounters
          INNER JOIN 
          --Subquery to get id, started, stopped from clinical_problems 
          (SELECT clinical_problems.patient_id, clinical_problems.started, clinical_problems.stopped  
          FROM clinical_problems)
          AS problems 
          ON problems.started < encounters."dateTime";
# with start dates for problems
  SELECT patient_id, name, started, stopped
  FROM clinical_problems
  where  started IS NOT NULL and patient_id IN (
    SELECT DISTINCT patient_id
    FROM  clinical_encounters 
    WHERE (lower(notes) like ('%emergency depart%')) or 
    (lower(notes) like lower(('%recent hospitalization%'))) or 
    (lower(notes) like lower(('% ER f%'))) or
    (lower(notes) like lower(('%discharge summary%'))) or  
    (lower(notes) like lower(('%hospital follow%'))) or
    (lower(notes) like lower(('%hosp follow%'))));

# medications for hospitalized patients. _note will likely want to stem
SELECT patient_id, generic, stopped
  FROM clinical_medications
  where generic is NOT NULL and generic NOT like('') and patient_id IN (
    SELECT DISTINCT patient_id
    FROM  clinical_encounters 
    WHERE (lower(notes) like ('%emergency depart%')) or 
    (lower(notes) like lower(('%recent hospitalization%'))) or 
    (lower(notes) like lower(('% ER f%'))) or
    (lower(notes) like lower(('%discharge summary%'))) or  
    (lower(notes) like lower(('%hospital follow%'))) or
    (lower(notes) like lower(('%hosp follow%'))));


# _Reference:The number of patients who have been to the ER or hospitalized = 492
select count(DISTINCT patient_id) 
  from clinical_encounters 
 where (notes like ('% ER%')) or (lower(notes) like lower(('%hospitalization%')));

 # _Reference: The number of patients who have not been to the ER or hospitalized 
select count(DISTINCT patient_id) 
  from clinical_encounters 
 where NOT((notes like ('% ER%')) or (lower(notes) like lower(('%hospitalization%'))) 
 AND (notes IS NOT NULL)); # taking null notes into account 

# _Reference: patinets who have been hospitalized and have document_id
select patient_id, document_id
  from clinical_encounters 
 where document_id IS NOT NULL AND ((notes like ('% ER%')) or (lower(notes) like lower(('%hospitalization%'))));

# _Reference: dates included for the search for hospitalizations and ER visits
select patient_id, notes, "dateTime"
  from clinical_encounters 
 where (lower(notes) like lower(('% emergency %')) or (lower(notes) like ('%recent hospitalization%')));




# _Reference: the total number of null entries for notes in clinical encounters
 select count(DISTINCT patient_id) 
  from clinical_encounters 
 where notes IS NOT NULL;


# _Reference: The number of patients who have not been hospitalized = 2923
select count(DISTINCT patient_id) 
  from clinical_encounters 
 where NOT((notes like ('% ER%')) or (lower(notes) like lower(('%hospitalization%'))) 
 AND (notes IS NOT NULL));
 

select DISTINCT patient_id, notes 
  from clinical_encounters 
 where NOT((notes like ('% ER%')) or (lower(notes) like lower(('%hospitalization%'))) 
 AND (notes IS NOT NULL));

 select DISTINCT patient_id, notes, "dateTime" 
  from clinical_encounters 
 where (notes NOT like ('% ER%')) and (lower(notes) NOT like lower(('%hospitalization%'))) 
 AND (notes IS NOT NULL) and (lower(notes) NOT like lower(('%CHART NOT SIGNED%'))) limit 1000;


# subquerry self join to find patients not hospitalized 
SELECT DISTINCT 
    B.patient_id AS unhospitalized_pts 
FROM 
   clinical_encounters B
WHERE 
    B.patient_id 
        NOT IN (
        SELECT A.patient_id 
        FROM clinical_encounters A
        WHERE (A.notes like ('% ER%')) or (lower(A.notes) like lower(('%hospitalization%')))
        AND (B.patient_id != A.patient_id));



 SELECT A.patient_id AS hospitalized_pts, 
 B.patient_id AS unhospitalized_pts
FROM clinical_encounters A, clinical_encounters B
WHERE (A.notes like ('% ER%')) or (lower(A.notes) like lower(('%hospitalization%')))
AND (B.notes NOT like ('% ER%')) and (B.lower(notes) NOT like lower(('%hospitalization%'))) 
 AND (B.notes IS NOT NULL) and (lower(B.notes) NOT like lower(('%CHART NOT SIGNED%')))
 AND A.patient_id <> B.patient.id  
ORDER BY B.patient_id;


SELECT  A.patient_id AS hospitalized_pts, 
 B.patient_id AS unhospitalized_pts 
FROM clinical_encounters A, clinical_encounters B
WHERE (A.notes like ('% ER%')) or (lower(A.notes) like lower(('%hospitalization%')))
AND (B.notes NOT like ('% ER%')) and (lower(B.notes) NOT like lower(('%hospitalization%'))) 
AND (B.notes IS NOT NULL) and (lower(B.notes) NOT like lower(('%CHART NOT SIGNED%')))
AND (A.patient_id != B.patient_id) limit 100;