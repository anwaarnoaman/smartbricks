import gradio as gr
import requests

# Function to call the rent estimation API
def estimate_rent(area, near_to_metro, no_of_rooms):
    api_url  = "http://flask-api:5000/estimate_rent"  # Replace with your actual API endpoint
    payload = {
        "area": area,
        "near_to_metro": near_to_metro,
        "no_of_rooms": no_of_rooms
    }
    try:
        response = requests.post(api_url, json=payload)
        response.raise_for_status()  # Raise an error for bad HTTP responses
        rent_estimation = response.json().get("estimated_rent", "Error in response")
        return f"Estimated Rent: {rent_estimation}"
    except Exception as e:
        return f"Error: {str(e)}"

# Define the Gradio interface
with gr.Blocks() as rent_estimation_app:
    gr.Markdown("# Property Rent Estimation App")
    with gr.Row():
        area_input = gr.Number(label="Area (in square feet)", value=500)
        near_to_metro_input = gr.Checkbox(label="Near to Metro?", value=False)
        no_of_rooms_input = gr.Number(label="Number of Rooms", value=2)

    estimate_button = gr.Button("Estimate Rent")
    rent_output = gr.Textbox(label="Rent Estimation Output")

    # Event listener for the button click
    estimate_button.click(
        fn=estimate_rent,
        inputs=[area_input, near_to_metro_input, no_of_rooms_input],
        outputs=[rent_output]
    )

# Launch the app
if __name__ == "__main__":
    rent_estimation_app.launch()
