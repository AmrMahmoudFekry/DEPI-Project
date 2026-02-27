import streamlit as st
from system import HospitalSystem
import pandas as pd

# Page Configuration
st.set_page_config(page_title="Hospital Management System", layout="wide")

# Custom CSS for better UI
st.markdown("""
    <style>
        .main { background-color: #f8f9fa; }
        .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; }
        .stDataFrame { border: 1px solid #e6e9ef; border-radius: 10px; }
    </style>
""", unsafe_allow_html=True)

# Initialize System
if "system" not in st.session_state:
    st.session_state.system = HospitalSystem()

system = st.session_state.system

st.title("🏥 Hospital Management System")
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3306/3306567.png", width=100)
menu = st.sidebar.selectbox(
    "Navigation Menu",
    ["View Records", "Add Patient", "Add Doctor", "Create Appointment"]
)

# ---------------- 1. VIEW DATA (Dashboard) ----------------
if menu == "View Records":
    st.subheader("📊 Hospital Database Records")
    
    tab1, tab2, tab3 = st.tabs(["Patients", "Doctors", "Appointments"])

    with tab1:
        if system.patients:
            st.dataframe(pd.DataFrame(system.patients), use_container_width=True)
        else:
            st.info("No patient records found.")

    with tab2:
        if system.doctors:
            st.dataframe(pd.DataFrame(system.doctors), use_container_width=True)
        else:
            st.info("No doctor records found.")

    with tab3:
        if system.appointments:
            st.dataframe(pd.DataFrame(system.appointments), use_container_width=True)
        else:
            st.info("No appointments scheduled.")

# ---------------- 2. ADD PATIENT ----------------
elif menu == "Add Patient":
    st.subheader("📝 Register New Patient")
    with st.form("patient_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        pid = col1.text_input("Patient ID (e.g., P004)")
        name = col2.text_input("Full Name")
        age = col1.number_input("Age", min_value=1, step=1)
        gender = col2.selectbox("Gender", ["Male", "Female", "Other"])
        disease = st.text_input("Current Disease/Diagnosis")
        
        submit = st.form_submit_button("Save Patient Record")
        if submit:
            if pid and name:
                try:
                    system.add_patient(pid, name, age, gender, disease)
                    st.success(f"Patient {name} registered successfully!")
                    st.balloons()
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.warning("Please fill in the ID and Name fields.")

# ---------------- 3. ADD DOCTOR (FIXED & LINKED) ----------------
elif menu == "Add Doctor":
    st.subheader("👨‍⚕️ Register New Medical Staff")
    with st.form("doctor_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        did = col1.text_input("Doctor ID (e.g., D004)")
        name = col2.text_input("Doctor Name")
        age = col1.number_input("Age", min_value=25, step=1)
        gender = col2.selectbox("Gender", ["Male", "Female"])
        spec = st.text_input("Specialization (e.g., Surgery)")
        
        submit = st.form_submit_button("Save Doctor Record")
        if submit:
            if did and name:
                try:
                    system.add_doctor(did, name, age, gender, spec)
                    st.success(f"Dr. {name} added to the system.")
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.warning("Doctor ID and Name are required.")

# ---------------- 4. CREATE APPOINTMENT ----------------
elif menu == "Create Appointment":
    st.subheader("📅 Schedule New Appointment")
    if not system.patients or not system.doctors:
        st.error("You need at least one patient and one doctor to create an appointment.")
    else:
        with st.form("appointment_form", clear_on_submit=True):
            aid = st.text_input("Appointment ID (e.g., A003)")
            
            # Select from existing IDs
            p_list = [p['ID'] for p in system.patients]
            d_list = [d['ID'] for d in system.doctors]
            
            pid = st.selectbox("Select Patient ID", p_list)
            did = st.selectbox("Select Doctor ID", d_list)
            date = st.date_input("Appointment Date")
            
            if st.form_submit_button("Confirm Booking"):
                try:
                    system.create_appointment(aid, pid, did, str(date))
                    st.success("Appointment booked successfully!")
                except Exception as e:
                    st.error(f"Error: {e}")