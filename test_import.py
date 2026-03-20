try:
    from lasercom_digital_twin.analysis.feedback_linearization import NdobFbl
    print("SUCCESS: Imported NdobFbl")
    model = NdobFbl()
    print("SUCCESS: Instantiated NdobFbl")
except Exception as e:
    import traceback
    traceback.print_exc()
