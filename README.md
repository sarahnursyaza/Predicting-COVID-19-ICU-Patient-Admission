# Predicting-COVID-19-ICU-Patient-Admission

## Assignment Step 3

### Introduction

The sudden and rapid growth of COVID-19 cases is overwhelming health systems globally with a demand for ICU beds far above the existing capacity. Thus, there is urgency in obtaining an accurate system to prioritize COVID-19 patients according to their health condition especially those that doesn't show any symptoms. Vital signs and blood test can help in determining their exact health.

### Objectives

    1.To predict admission to the ICU of confirmed COVID-19 cases based on clinical test and vital signs.
    2.To improve medical decision-making by prioritizing COVID-19 patients in need of intensive care.


### Dataset

Kaggle: https://www.kaggle.com/S%C3%ADrio-Libanes/covid19

This dataset contains anonymized data from Hospital Sírio-Libanês, São Paulo and Brasilia. All data were anonymized following the best international practices and recommendations. Data has been cleaned and scaled by column according to Min Max Scaler to fit between -1 and 1.

Each patient health was recorded based on the duration of hospitalization (Table 1) below:

<table class="table table-bordered">
    <thead>
        <tr>
            <th>Window</th>
            <th>Description</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>0-2</td>
            <td>From 0 to 2 hours of the admission</td>
        </tr>
        <tr>
            <td>2-4</td>
            <td>From 2 to 4 hours of the admission</td>
        </tr>
        <tr>
            <td>4-6</td>
            <td>From 4 to 6 hours of the admission</td>
        </tr>
        <tr>
            <td>6-12</td>
            <td>From 6 to 12 hours of the admission</td>
        </tr>
        <tr>
            <td>Above-12</td>
            <td>Above 12 hours from admission</td>
        </tr>
    </tbody>
</table>

Available data:

    a. Patient demographic information
    b. Patient previous grouped diseases
    c. Blood results
    d. Vital signs
    e. Blood gases

In total there are 42 features, expanded to the mean, max, min, diff and relative diff.
