from Classes import Patient, Doctor, Appointment
from file_handling import StorageManager

class HospitalSystem:
    def __init__(self):
        self.file_manager = StorageManager()
        self.patients = self.load_patients()
        self.doctors = self.load_doctors()
        self.appointments = self.load_appointments()

    def load_patients(self):
        data = []
        lines = self.file_manager.read_all("patients.txt")
        for line in lines:
            parts = line.strip().split("|")
            if len(parts) == 5:
                data.append({"ID": parts[0], "Name": parts[1], "Age": parts[2], "Gender": parts[3], "Disease": parts[4]})
        return data

    def load_doctors(self):
        data = []
        lines = self.file_manager.read_all("doctors.txt")
        for line in lines:
            parts = line.strip().split("|")
            if len(parts) == 5:
                data.append({"ID": parts[0], "Name": parts[1], "Age": parts[2], "Gender": parts[3], "Specialty": parts[4]})
        return data

    def load_appointments(self):
        data = []
        lines = self.file_manager.read_all("appointments.txt")
        for line in lines:
            parts = line.strip().split("|")
            if len(parts) == 4:
                data.append({"Appt ID": parts[0], "Patient ID": parts[1], "Doctor ID": parts[2], "Date": parts[3]})
        return data

    def add_patient(self, pid, name, age, gender, disease):
        new_p = Patient(pid, name, int(age), gender, disease)
        self.file_manager.append("patients.txt", new_p.to_line())
        self.patients.append({"ID": pid, "Name": name, "Age": age, "Gender": gender, "Disease": disease})

    def add_doctor(self, did, name, age, gender, specialty):
        new_d = Doctor(did, name, int(age), gender, specialty)
        self.file_manager.append("doctors.txt", new_d.to_line())
        self.doctors.append({"ID": did, "Name": name, "Age": age, "Gender": gender, "Specialty": specialty})

    def create_appointment(self, aid, pid, did, date):
        new_a = Appointment(aid, pid, did, date)
        self.file_manager.append("appointments.txt", new_a.to_line())
        self.appointments.append({"Appt ID": aid, "Patient ID": pid, "Doctor ID": did, "Date": date})