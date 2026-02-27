from datetime import datetime

class Patient:
    def __init__(self, patient_id, name, age, gender, disease):
        if not isinstance(age, int) or age <= 0:
            try: age = int(age)
            except: raise ValueError("Age must be a positive integer.")

        self.patient_id = patient_id
        self.name = name
        self.age = age
        self.gender = gender
        self.disease = disease

    def to_line(self):
        return f"{self.patient_id}|{self.name}|{self.age}|{self.gender}|{self.disease}"

class Doctor:
    def __init__(self, doctor_id, name, age, gender, specialty):
        self.doctor_id = doctor_id
        self.name = name
        self.age = age
        self.gender = gender
        self.specialty = specialty

    def to_line(self):
        return f"{self.doctor_id}|{self.name}|{self.age}|{self.gender}|{self.specialty}"

class Appointment:
    def __init__(self, appointment_id, patient_id, doctor_id, appointment_date):
        self.appointment_id = appointment_id
        self.patient_id = patient_id
        self.doctor_id = doctor_id
        self.appointment_date = appointment_date

    def to_line(self):
        return f"{self.appointment_id}|{self.patient_id}|{self.doctor_id}|{self.appointment_date}"