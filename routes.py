from flask import Blueprint, render_template, request, jsonify, redirect, url_for, flash
from flask_login import login_required, current_user
from extensions import db
from models import User, Prediction
from predict import predict_price, get_cities_locations, get_model_metrics, get_meta, diagnose_location

main_bp = Blueprint('main', __name__)


@main_bp.route('/')
def index():
    return redirect(url_for('auth.login'))


@main_bp.route('/dashboard')
@login_required
def dashboard():
    total_predictions = Prediction.query.filter_by(user_id=current_user.id).count()
    recent = (Prediction.query
              .filter_by(user_id=current_user.id)
              .order_by(Prediction.created_at.desc())
              .limit(5).all())
    metrics = get_model_metrics()
    meta    = get_meta()
    return render_template('dashboard.html',
                           total_predictions=total_predictions,
                           recent=recent,
                           metrics=metrics,
                           meta=meta)


@main_bp.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    # Always pass cities_locations — same name used in both route and template
    cities_locations = get_cities_locations()

    if request.method == 'POST':
        city     = request.form.get('city', '').strip().lower()
        location = request.form.get('location', '').strip()
        sqft     = request.form.get('total_sqft', 0, type=float)
        bhk      = request.form.get('bhk', 2, type=int)
        bath     = request.form.get('bath', 2, type=int)

        if not city or not location or sqft <= 0:
            flash('Please fill in all fields correctly.', 'danger')
            return render_template('predict.html', cities_locations=cities_locations)

        try:
            result = predict_price(city, location, sqft, bhk, bath)

            pred = Prediction(
                user_id=current_user.id,
                location=f"{city.capitalize()} — {location}",
                total_sqft=sqft, bhk=bhk, bath=bath,
                predicted_price=result['price']
            )
            db.session.add(pred)
            db.session.commit()

            return render_template('result.html', result=result)

        except RuntimeError as e:
            flash(str(e), 'danger')
            return render_template('predict.html', cities_locations=cities_locations)

    return render_template('predict.html', cities_locations=cities_locations)


@main_bp.route('/history')
@login_required
def history():
    page = request.args.get('page', 1, type=int)
    predictions = (Prediction.query
                   .filter_by(user_id=current_user.id)
                   .order_by(Prediction.created_at.desc())
                   .paginate(page=page, per_page=10))
    return render_template('history.html', predictions=predictions)


# ── REST API ──────────────────────────────────────────────────────────────────

@main_bp.route('/api/predict', methods=['POST'])
@login_required
def api_predict():
    data = request.get_json(force=True)
    try:
        result = predict_price(
            data['city'],
            data['location'],
            float(data['total_sqft']),
            int(data['bhk']),
            int(data['bath'])
        )
        return jsonify({'status': 'success', 'data': result})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400


@main_bp.route('/api/locations')
@login_required
def api_locations():
    return jsonify(get_cities_locations())


@main_bp.route('/api/locations/<city>')
@login_required
def api_locations_city(city):
    cl   = get_cities_locations()
    locs = cl.get(city.lower(), [])
    return jsonify({'city': city, 'locations': locs})


# ── DEBUG ROUTE (remove in production) ───────────────────────────────────────
@main_bp.route('/debug/location')
@login_required
def debug_location():
    """Usage: /debug/location?city=chennai&location=adyar"""
    city     = request.args.get('city', 'chennai')
    location = request.args.get('location', 'adyar')
    return jsonify(diagnose_location(city, location))