from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/estimate_rent', methods=['POST'])
def estimate_rent():
    data = request.json
    try:
        # Extract inputs
        area = data.get("area")
        near_to_metro = data.get("near_to_metro")
        no_of_rooms = data.get("no_of_rooms")

        if not all([area, no_of_rooms]) or not isinstance(near_to_metro, bool):
            return jsonify({"error": "Invalid input"}), 400

        # Dummy estimation logic
        base_rent = 10  # Rent per square foot
        metro_bonus = 200 if near_to_metro else 0
        room_multiplier = 50 * no_of_rooms

        estimated_rent = (area * base_rent) + metro_bonus + room_multiplier

        return jsonify({"estimated_rent": estimated_rent})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
