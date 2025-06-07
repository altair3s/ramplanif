
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit.components.v1 as components
import io
import tempfile
import os
from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import warnings

warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="Système de Planification Ramp Handling",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(90deg, #f0f2f6, #ffffff);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .qualification-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        margin: 0.2rem;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.8rem;
    }
    .cz-badge { background-color: #ff6b6b; color: white; }
    .mop-badge { background-color: #4ecdc4; color: white; }
    .push-badge { background-color: #45b7d1; color: white; }
    .agent-badge { background-color: #96ceb4; color: white; }
</style>
""", unsafe_allow_html=True)


# Cache pour optimiser les performances
@st.cache_data
def cached_preprocess_data(data):
    return preprocess_data(data)


class RampPlanningSystem:
    def __init__(self):
        # Structure des zones basée sur votre schéma
        self.zones = {
            "Zone_1": ["69", "70", "71", "72"],
            "Zone_2": ["66", "67", "67B", "68"],
            "Zone_3": ["63", "64", "65"],
            "Zone_4": ["60", "61", "62"],
            "Zone_5": ["58", "59"],  # Ajusté selon le schéma
            "Zone_6": ["51", "51B", "52", "53", "54", "55", "55B", "56", "57", "57B"]
        }

        self.qualifications = ["CZ", "MOP", "PUSH", "AGENT"]
        self.vacation_shifts = {
            "Matin": {"start": time(4, 30), "end": time(14, 0)},
            "Soir": {"start": time(13, 45), "end": time(23, 15)}
        }

        # Contraintes légales belges
        self.legal_constraints = {
            "weekly_hours": 38,
            "daily_max_hours": 10,
            "min_rest_hours": 11,
            "weekly_rest_hours": 35,
            "monthly_max_hours": 175,
            "voluntary_monthly_max": 180
        }


def parse_time(time_value, date=None):
    """Parse les heures en format datetime"""
    if pd.isna(time_value):
        return pd.NaT
    if date is None:
        date = datetime.today().date()

    try:
        if isinstance(time_value, str):
            # Essayer différents formats
            for fmt in ['%H:%M:%S', '%H:%M', '%H:%M:%S.%f']:
                try:
                    parsed_time = datetime.strptime(time_value, fmt).time()
                    return datetime.combine(date, parsed_time)
                except ValueError:
                    continue
            return pd.NaT
        elif isinstance(time_value, datetime):
            return datetime.combine(date, time_value.time())
        elif isinstance(time_value, timedelta):
            base_time = (datetime.min + time_value).time()
            return datetime.combine(date, base_time)
        elif isinstance(time_value, time):
            return datetime.combine(date, time_value)
        elif isinstance(time_value, (int, float)):
            # Gérer les temps en format décimal (ex: 14.5 = 14:30)
            hours = int(time_value)
            minutes = int((time_value - hours) * 60)
            parsed_time = time(hours, minutes)
            return datetime.combine(date, parsed_time)
    except Exception:
        return pd.NaT

    return pd.NaT


def preprocess_data(data):
    """Traite et nettoie les données de vol"""
    df = data.copy()

    # Mapping des colonnes selon le format réel du fichier
    column_mapping = {
        'Date': 'DATE',
        'n°arr': 'VOLA',
        'n°dep': 'VOLD',
        'Origine': 'ORG',
        'Destination': 'DEST',
        'Arr': 'HA',
        'Dép': 'HD',
        'Parking': 'STAND',
        'SO': 'PAX'
    }

    # Renommer les colonnes selon le mapping
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns:
            df[new_col] = df[old_col]

    # Créer la colonne ZONE basée sur le parking
    def determine_zone(parking):
        """Détermine la zone basée sur le numéro de parking"""
        try:
            parking_str = str(parking).strip()
            if parking_str in ["69", "70", "71", "72"]:
                return "Zone_1"
            elif parking_str in ["66", "67", "67B", "68"]:
                return "Zone_2"
            elif parking_str in ["63", "64", "65"]:
                return "Zone_3"
            elif parking_str in ["60", "61", "62"]:
                return "Zone_4"
            elif parking_str in ["58", "59"]:
                return "Zone_5"
            elif parking_str in ["51", "51B", "52", "53", "54", "55", "55B", "56", "57", "57B"]:
                return "Zone_6"
            else:
                return "Zone_1"  # Zone par défaut
        except:
            return "Zone_1"

    df['ZONE'] = df['STAND'].apply(determine_zone)

    # S'assurer que les colonnes requises existent
    required_columns = ['DATE', 'VOLA', 'VOLD', 'ORG', 'DEST', 'HA', 'HD', 'STAND', 'PAX', 'ZONE']
    for col in required_columns:
        if col not in df.columns:
            st.warning(f"La colonne {col} est manquante dans le fichier")
            # Créer des colonnes manquantes avec des valeurs par défaut
            if col == 'ZONE':
                df[col] = 'Zone_1'
            elif col in ['PAX']:
                df[col] = 0
            else:
                df[col] = ''

    # Convertir la colonne DATE en datetime
    try:
        df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
    except Exception as e:
        st.error(f"Erreur lors de la conversion de la colonne DATE : {e}")
        return pd.DataFrame()

    # Fonction pour parser les heures au format "0620" -> 06:20
    def parse_time_format(time_str, date=None):
        """Parse les heures du format '0620' vers datetime"""
        if pd.isna(time_str) or time_str == '-' or str(time_str).strip() == '':
            return pd.NaT

        if date is None:
            date = datetime.today().date()

        try:
            time_str = str(time_str).strip()

            # Si c'est déjà au format HH:MM ou HH:MM:SS
            if ':' in time_str:
                for fmt in ['%H:%M:%S', '%H:%M']:
                    try:
                        parsed_time = datetime.strptime(time_str, fmt).time()
                        return datetime.combine(date, parsed_time)
                    except ValueError:
                        continue

            # Format "0620" -> 06:20
            if len(time_str) == 4 and time_str.isdigit():
                hours = int(time_str[:2])
                minutes = int(time_str[2:])
                if 0 <= hours <= 23 and 0 <= minutes <= 59:
                    parsed_time = time(hours, minutes)
                    return datetime.combine(date, parsed_time)

            # Format "620" -> 06:20 (sans le 0 initial)
            if len(time_str) == 3 and time_str.isdigit():
                hours = int(time_str[0])
                minutes = int(time_str[1:])
                if 0 <= hours <= 23 and 0 <= minutes <= 59:
                    parsed_time = time(hours, minutes)
                    return datetime.combine(date, parsed_time)

        except Exception:
            pass

        return pd.NaT

    # Convertir les colonnes HA et HD en datetime
    try:
        df['HA'] = df.apply(
            lambda row: parse_time_format(row['HA'], row['DATE'].date()) if pd.notna(row['DATE']) else pd.NaT, axis=1)
        df['HD'] = df.apply(
            lambda row: parse_time_format(row['HD'], row['DATE'].date()) if pd.notna(row['DATE']) else pd.NaT, axis=1)

        # S'assurer que les colonnes sont bien de type datetime
        df['HA'] = pd.to_datetime(df['HA'], errors='coerce')
        df['HD'] = pd.to_datetime(df['HD'], errors='coerce')

    except Exception as e:
        st.error(f"Erreur lors de la conversion des heures : {e}")
        return pd.DataFrame()

    # Gérer les night stops (ajouter 30 minutes à HA si HD manque)
    night_stop_mask = df['HA'].notna() & df['HD'].isna()
    if night_stop_mask.any():
        df.loc[night_stop_mask, 'HD'] = df.loc[night_stop_mask, 'HA'] + timedelta(minutes=30)

    # Gérer les départs secs (soustraire 30 minutes de HD si HA manque)
    depart_sec_mask = df['HD'].notna() & df['HA'].isna()
    if depart_sec_mask.any():
        df.loc[depart_sec_mask, 'HA'] = df.loc[depart_sec_mask, 'HD'] - timedelta(minutes=30)

    # Extraction de la compagnie
    def extract_company(row):
        try:
            if pd.notna(row['VOLD']) and str(row['VOLD']).strip() != '' and str(row['VOLD']) != '-':
                company = str(row['VOLD']).split()[0]
                return ''.join(filter(str.isalpha, company)) if company else None
            elif pd.notna(row['VOLA']) and str(row['VOLA']).strip() != '' and str(row['VOLA']) != '-':
                company = str(row['VOLA']).split()[0]
                return ''.join(filter(str.isalpha, company)) if company else None
            return 'Unknown'
        except Exception:
            return 'Unknown'

    df['Company'] = df.apply(extract_company, axis=1)

    # Calculer la durée
    try:
        duration_mask = df['HA'].notna() & df['HD'].notna()
        df.loc[duration_mask, 'Duration'] = (df.loc[duration_mask, 'HD'] - df.loc[
            duration_mask, 'HA']).dt.total_seconds() / 60
        df['Duration'] = df['Duration'].fillna(0)
    except Exception as e:
        st.warning(f"Erreur lors du calcul de la durée : {e}")
        df['Duration'] = 0

    # S'assurer que STAND et PAX sont numériques
    df['STAND'] = pd.to_numeric(df['STAND'], errors='coerce').fillna(0)
    df['PAX'] = pd.to_numeric(df['PAX'], errors='coerce').fillna(0)

    # Filtrer les lignes qui ont soit HA soit HD
    valid_mask = df['HA'].notna() | df['HD'].notna()
    result_df = df[valid_mask].copy()

    if len(result_df) == 0:
        st.warning("Aucune donnée valide trouvée après traitement")
    else:
        st.success(f"Données traitées avec succès : {len(result_df)} vols trouvés")

    return result_df


def assign_vacation_lines_by_zone(data, vacation_amplitude_hours=8, min_gap_minutes=10):
    """Assigne les vols aux lignes de vacation par zone avec gestion stricte des chevauchements"""
    df = data.copy()

    # Filtrer seulement les lignes avec des heures valides
    valid_times = df['HA'].notna() & df['HD'].notna()
    if not valid_times.any():
        st.warning("Aucune donnée avec heures valides pour assigner les vacations")
        df['Vacation Line'] = 0
        df['Zone_Vacation'] = df['ZONE'] + '_V0'
        return df

    df = df[valid_times].copy()
    df['Vacation Line'] = 0
    df['Zone_Vacation'] = ''

    min_gap = timedelta(minutes=min_gap_minutes)

    # Traiter chaque zone séparément
    for zone in df['ZONE'].unique():
        zone_mask = df['ZONE'] == zone
        zone_data = df[zone_mask].copy()
        zone_data = zone_data.sort_values('HA').reset_index()

        # Initialiser toutes les lignes de vacation à -1 (non assigné)
        zone_data['temp_line'] = -1

        # Liste pour stocker les intervalles occupés par ligne de vacation
        # Format: ligne_number -> [(start1, end1), (start2, end2), ...]
        line_intervals = {}

        # Assigner chaque vol à une ligne
        for idx, flight in zone_data.iterrows():
            flight_start = flight['HA']
            flight_end = flight['HD']

            assigned_line = None

            # Chercher une ligne existante où le vol peut s'insérer
            for line_num in sorted(line_intervals.keys()):
                can_fit = True

                # Vérifier tous les intervalles de cette ligne
                for interval_start, interval_end in line_intervals[line_num]:
                    # Vérifier s'il y a conflit avec cet intervalle
                    # Un vol peut s'insérer s'il se termine avant le début d'un intervalle
                    # ou commence après la fin d'un intervalle (avec marge)
                    if not (flight_end + min_gap <= interval_start or
                            flight_start >= interval_end + min_gap):
                        can_fit = False
                        break

                if can_fit:
                    assigned_line = line_num
                    break

            # Si aucune ligne existante ne convient, créer une nouvelle ligne
            if assigned_line is None:
                assigned_line = len(line_intervals)
                line_intervals[assigned_line] = []

            # Ajouter cet intervalle à la ligne assignée
            line_intervals[assigned_line].append((flight_start, flight_end))
            zone_data.loc[idx, 'temp_line'] = assigned_line

        # Trier les intervalles de chaque ligne pour optimiser les futures vérifications
        for line_num in line_intervals:
            line_intervals[line_num].sort()

        # Mettre à jour le DataFrame principal avec les indices originaux
        for idx, row in zone_data.iterrows():
            original_idx = row['index']  # L'index original conservé lors du reset_index
            df.loc[original_idx, 'Vacation Line'] = row['temp_line']
            df.loc[original_idx, 'Zone_Vacation'] = f"{zone}_V{row['temp_line']}"

    return df


def create_interactive_gantt(data, selected_date, system):
    """Crée un diagramme de Gantt interactif organisé par zone avec gestion des chevauchements"""
    # Filtrer les données pour la date sélectionnée
    try:
        date_mask = pd.to_datetime(data['DATE']).dt.date == selected_date.date()
        data = data[date_mask].copy()
    except Exception as e:
        st.error(f"Erreur lors du filtrage par date : {e}")
        return None, None, 0

    if len(data) == 0:
        st.warning("Aucune donnée pour cette date")
        return None, None, 0

    col1, col2 = st.columns(2)
    with col1:
        vacation_amplitude = st.sidebar.slider(
            "Amplitude des vacations (heures)",
            min_value=4,
            max_value=12,
            value=8,
            step=1,
            key="slider_amplitude"
        )
    with col2:
        st.write("")

    col3, col4 = st.columns(2)
    with col3:
        min_gap = st.sidebar.slider(
            "Écart minimum entre les tâches (minutes)",
            min_value=5,
            max_value=30,
            value=10,
            step=5,
            key="slider_unique_key"
        )
    with col4:
        st.write("")

    # Assigner les lignes de vacation par zone avec gestion des chevauchements
    data = assign_vacation_lines_by_zone(data, vacation_amplitude, min_gap)

    # Calculer le nombre de lignes de vacation créées
    total_vacations = data['Zone_Vacation'].nunique()

    # Générer le graphique Gantt organisé par zone
    try:
        # S'assurer que les colonnes datetime sont correctes
        data['HA'] = pd.to_datetime(data['HA'], errors='coerce')
        data['HD'] = pd.to_datetime(data['HD'], errors='coerce')

        # Créer les strings pour affichage
        data['HA_str'] = data['HA'].dt.strftime('%H:%M:%S')
        data['HD_str'] = data['HD'].dt.strftime('%H:%M:%S')

    except Exception as e:
        st.error(f"Erreur lors de la conversion des heures : {e}")
        return None, None, 0

    data['Flight_Type'] = 'Normal'
    data.loc[pd.isna(data['VOLA']) & pd.notna(data['VOLD']), 'Flight_Type'] = 'Depart_Sec'
    data.loc[pd.notna(data['VOLA']) & pd.isna(data['VOLD']), 'Flight_Type'] = 'Night_Stop'

    def annotation(row):
        try:
            if pd.notna(row['VOLD']) and pd.notna(row['VOLA']):
                return f"{row['VOLD']} ({row['STAND']})"
            elif pd.isna(row['VOLD']):
                return f"{row['VOLA']} ({row['STAND']})"
            else:
                return f"{row['VOLD']} ({row['STAND']})"
        except Exception:
            return f"Vol ({row['STAND']})"

    data['Annotation'] = data.apply(annotation, axis=1)

    # Filtrer les données avec des heures valides pour le graphique
    valid_data = data[(data['HA'].notna()) & (data['HD'].notna())].copy()

    if len(valid_data) == 0:
        st.warning("Aucune donnée avec heures valides pour créer le graphique")
        return None, None, 0

    # Créer des sous-graphiques par zone
    zones = sorted(valid_data['ZONE'].unique())

    from plotly.subplots import make_subplots

    # Calculer la hauteur nécessaire pour chaque zone
    zone_heights = {}
    for zone in zones:
        zone_data = valid_data[valid_data['ZONE'] == zone]
        max_vacation_line = zone_data['Vacation Line'].max() if len(zone_data) > 0 else 0
        zone_heights[zone] = max(max_vacation_line + 1, 1)  # Au moins 1 ligne

    total_height = sum(zone_heights.values())

    fig = make_subplots(
        rows=len(zones),
        cols=1,
        subplot_titles=[f"{zone} - {zone_heights[zone]} ligne(s) de vacation" for zone in zones],
        vertical_spacing=0.03,
        shared_xaxes=True,
        row_heights=[zone_heights[zone] / total_height for zone in zones]
    )

    # Couleurs par zone
    zone_colors = {
        'Zone_1': '#FF9999', 'Zone_2': '#66B2FF', 'Zone_3': '#99FF99',
        'Zone_4': '#FFCC99', 'Zone_5': '#FF99CC', 'Zone_6': '#99CCFF'
    }

    min_time = valid_data['HA'].min()
    max_time = valid_data['HD'].max()
    vacation_timedelta = timedelta(hours=vacation_amplitude)

    for i, zone in enumerate(zones, 1):
        zone_data = valid_data[valid_data['ZONE'] == zone].copy()

        if len(zone_data) == 0:
            continue

        # Créer les barres pour chaque vol dans cette zone
        for _, row in zone_data.iterrows():
            # Utiliser la ligne de vacation pour éviter les superpositions
            y_position = row['Vacation Line']

            fig.add_trace(
                go.Scatter(
                    x=[row['HA'], row['HD'], row['HD'], row['HA'], row['HA']],
                    y=[y_position, y_position, y_position + 0.8, y_position + 0.8, y_position],
                    fill='toself',
                    fillcolor=zone_colors.get(zone, '#CCCCCC'),
                    line=dict(color='rgb(8,48,107)', width=1.5),
                    name=f"{zone}",
                    showlegend=False,
                    hovertemplate=f"<b>{row['Annotation']}</b><br>" +
                                  f"Zone: {zone}<br>" +
                                  f"Ligne: {row['Vacation Line']}<br>" +
                                  f"Début: {row['HA_str']}<br>" +
                                  f"Fin: {row['HD_str']}<br>" +
                                  f"Type: {row['Flight_Type']}<br>" +
                                  f"PAX: {row['PAX']}<br>" +
                                  "<extra></extra>"
                ),
                row=i, col=1
            )

            # Ajouter les annotations pour chaque vol
            fig.add_annotation(
                x=row['HA'] + (row['HD'] - row['HA']) / 2,
                y=y_position + 0.4,
                text=row['Annotation'],
                showarrow=False,
                font=dict(size=9, color="black"),
                align="center",
                xanchor="center",
                yanchor="middle",
                row=i, col=1
            )

            # Marquer les types de vols spéciaux
            if row['Flight_Type'] == 'Depart_Sec':
                fig.add_shape(
                    type="line",
                    x0=row['HD'],
                    x1=row['HD'],
                    y0=y_position,
                    y1=y_position + 0.8,
                    line=dict(color="#FF0000", width=4),
                    layer="above",
                    row=i, col=1
                )
            elif row['Flight_Type'] == 'Night_Stop':
                fig.add_shape(
                    type="line",
                    x0=row['HA'],
                    x1=row['HA'],
                    y0=y_position,
                    y1=y_position + 0.8,
                    line=dict(color="#334cff", width=4),
                    layer="above",
                    row=i, col=1
                )

        # Ajouter des lignes de séparation entre les vacations
        max_line = zone_data['Vacation Line'].max()
        for line_num in range(max_line + 1):
            if line_num > 0:  # Ligne de séparation
                fig.add_shape(
                    type="line",
                    x0=min_time,
                    x1=max_time,
                    y0=line_num - 0.1,
                    y1=line_num - 0.1,
                    line=dict(color="rgba(128, 128, 128, 0.3)", width=1, dash="dot"),
                    layer="below",
                    row=i, col=1
                )

    fig.update_layout(
        height=max(200 * len(zones), 400),
        width=1500,
        title_text=f"Planning des vols par zone - {selected_date.strftime('%d/%m/%Y')} - Amplitude: {vacation_amplitude}h - {total_vacations} lignes de vacation",
        plot_bgcolor='rgba(240, 248, 255, 0.8)',
        paper_bgcolor='white',
        margin=dict(l=50, r=50, t=80, b=50),
        showlegend=False
    )

    # Mettre à jour les axes
    for i, zone in enumerate(zones, 1):
        max_line = zone_heights[zone] - 1

        fig.update_yaxes(
            title_text="Ligne vacation",
            gridcolor='rgba(128, 128, 128, 0.2)',
            range=[-0.2, max_line + 1],
            tickmode='linear',
            tick0=0,
            dtick=1,
            row=i, col=1
        )

        if i == len(zones):  # Seulement pour le dernier graphique
            fig.update_xaxes(
                title_text="Heure",
                gridcolor='rgba(128, 128, 128, 0.2)',
                tickformat='%H:%M',
                row=i, col=1
            )
        else:
            fig.update_xaxes(
                gridcolor='rgba(128, 128, 128, 0.2)',
                tickformat='%H:%M',
                row=i, col=1
            )

    # Ajouter une légende explicative
    fig.add_annotation(
        x=1.02,
        y=1,
        xref="paper",
        yref="paper",
        text="<b>Légende:</b><br>" +
             "🔴 Liseret Rouge = Départ Sec<br>" +
             "🔵 Liseret Bleu = Night Stop<br>" +
             "📏 Chaque ligne = 1 vacation<br>" +
             "⚡ Vols simultanés = lignes séparées",
        showarrow=False,
        font=dict(size=10),
        align="left",
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="gray",
        borderwidth=1
    )

    return fig, data, total_vacations


def calculate_team_requirements_from_flights(data, selected_date, system):
    """Calcule les besoins en équipes basés sur les données de vol réelles (une équipe par zone par tranche)"""
    try:
        # Filtrer par date avec gestion d'erreur
        date_mask = pd.to_datetime(data['DATE']).dt.date == selected_date.date()
        date_flights = data[date_mask].copy()

        if len(date_flights) == 0:
            return {}

        # S'assurer que les colonnes datetime sont correctes
        date_flights['HA'] = pd.to_datetime(date_flights['HA'], errors='coerce')
        date_flights['HD'] = pd.to_datetime(date_flights['HD'], errors='coerce')

        # Filtrer seulement les vols avec des heures valides
        valid_flights = date_flights[(date_flights['HA'].notna()) | (date_flights['HD'].notna())].copy()

        if len(valid_flights) == 0:
            return {}

        # Assigner les lignes de vacation pour avoir une meilleure estimation
        valid_flights = assign_vacation_lines_by_zone(valid_flights, 8, 10)

        requirements = {}

        for shift_name, shift_times in system.vacation_shifts.items():
            try:
                shift_start = datetime.combine(selected_date.date(), shift_times["start"])
                shift_end = datetime.combine(selected_date.date(), shift_times["end"])

                if shift_end < shift_start:  # Vacation de nuit
                    shift_end += timedelta(days=1)

                # Filtrer les vols dans cette vacation
                shift_mask = (
                        (valid_flights['HA'] >= shift_start) &
                        (valid_flights['HA'] <= shift_end) &
                        (valid_flights['HA'].notna())
                )
                shift_flights = valid_flights[shift_mask]

                zone_requirements = {}
                for zone_name in system.zones.keys():
                    # Données pour cette zone dans cette vacation
                    zone_flights = shift_flights[shift_flights['ZONE'] == zone_name]

                    if len(zone_flights) > 0:
                        # UNE SEULE équipe par zone par tranche horaire
                        # Cette équipe traite tous les vols de la zone dans sa tranche
                        teams_needed = 1  # Une seule équipe par zone

                        zone_requirements[zone_name] = {
                            "teams": teams_needed,
                            "flights_count": len(zone_flights),
                            "qualifications": {
                                "CZ": teams_needed,
                                "MOP": teams_needed,
                                "PUSH": teams_needed,
                                "AGENT": teams_needed
                            }
                        }
                    else:
                        zone_requirements[zone_name] = {
                            "teams": 0,
                            "flights_count": 0,
                            "qualifications": {"CZ": 0, "MOP": 0, "PUSH": 0, "AGENT": 0}
                        }

                requirements[shift_name] = zone_requirements

            except Exception as e:
                st.warning(f"Erreur lors du calcul pour la vacation {shift_name} : {e}")
                continue

        return requirements

    except Exception as e:
        st.error(f"Erreur lors du calcul des besoins en équipes : {e}")
        return {}


def calculate_hours_and_etp_summary(data, selected_date, vacation_amplitude_hours=8):
    """Calcule la synthèse des heures et ETP"""
    try:
        # Filtrer par date
        date_mask = pd.to_datetime(data['DATE']).dt.date == selected_date.date()
        filtered_data = data[date_mask].copy()

        if len(filtered_data) == 0:
            return {}

        # Assigner les vacations par zone
        processed_data = assign_vacation_lines_by_zone(filtered_data, vacation_amplitude_hours, 10)

        summary = {}
        total_hours = 0
        total_etp = 0

        # Calculer pour chaque zone
        for zone in processed_data['ZONE'].unique():
            zone_data = processed_data[processed_data['ZONE'] == zone]

            # Nombre de vacations pour cette zone
            nb_vacations = len(zone_data['Vacation Line'].unique())

            # Heures totales pour cette zone (nombre de vacations × amplitude)
            zone_hours = nb_vacations * vacation_amplitude_hours

            # ETP pour cette zone (heures / 8h standard)
            zone_etp = zone_hours / 8

            summary[zone] = {
                'nb_vacations': nb_vacations,
                'heures_totales': zone_hours,
                'etp': round(zone_etp, 2),
                'nb_vols': len(zone_data)
            }

            total_hours += zone_hours
            total_etp += zone_etp

        summary['TOTAL'] = {
            'nb_vacations': sum(s['nb_vacations'] for s in summary.values()),
            'heures_totales': total_hours,
            'etp': round(total_etp, 2),
            'nb_vols': len(processed_data)
        }

        return summary

    except Exception as e:
        st.error(f"Erreur lors du calcul de la synthèse : {e}")
        return {}


def create_charge_curve(data, selected_date):
    """Crée la courbe de charge par tranche de 15 minutes avec des courbes en escalier"""
    try:
        # Filtrer par date avec une conversion robuste
        date_mask = pd.to_datetime(data['DATE']).dt.date == selected_date.date()
        filtered_data = data[date_mask].copy()

        if len(filtered_data) == 0:
            st.info("Aucune donnée pour cette date")
            return None

        # S'assurer que les colonnes sont en datetime
        filtered_data['HA'] = pd.to_datetime(filtered_data['HA'], errors='coerce')
        filtered_data['HD'] = pd.to_datetime(filtered_data['HD'], errors='coerce')

        # Filtrer les lignes avec des heures valides
        valid_ha = filtered_data['HA'].notna()
        valid_hd = filtered_data['HD'].notna()

        if not valid_ha.any() and not valid_hd.any():
            st.info("Aucune heure valide trouvée pour cette date")
            return None

        # Grouper par tranches de 15 minutes
        def group_by_15_minutes(time_col):
            return time_col.dt.floor('15T')

        # Traiter seulement les heures valides
        ha_data = filtered_data[valid_ha].copy()
        hd_data = filtered_data[valid_hd].copy()

        ha_counts = pd.Series([], dtype='int64')
        hd_counts = pd.Series([], dtype='int64')

        if len(ha_data) > 0:
            ha_data['HA_15min'] = group_by_15_minutes(ha_data['HA'])
            ha_counts = ha_data['HA_15min'].value_counts().sort_index()

        if len(hd_data) > 0:
            hd_data['HD_15min'] = group_by_15_minutes(hd_data['HD'])
            hd_counts = hd_data['HD_15min'].value_counts().sort_index()

        # Combiner les indices
        all_times = ha_counts.index.union(hd_counts.index)

        if len(all_times) == 0:
            st.info("Aucune tranche horaire valide trouvée")
            return None

        # Fusionner les comptages dans un DataFrame
        charge_curve = pd.DataFrame({
            'Tranche Horaire': all_times,
            'VOLA (HA)': ha_counts.reindex(all_times, fill_value=0),
            'VOLD (HD)': hd_counts.reindex(all_times, fill_value=0)
        }).reset_index(drop=True)

        # Créer un graphique en escalier avec Plotly
        fig = go.Figure()

        # Ajouter la courbe VOLA en escalier
        fig.add_trace(go.Scatter(
            x=charge_curve['Tranche Horaire'],
            y=charge_curve['VOLA (HA)'],
            mode='lines',
            line=dict(shape='hv', color='#FF0000', width=3),  # hv = horizontal puis vertical
            name='VOLA (Arrivées)',
            fill='tonexty' if 'VOLD (HD)' in charge_curve.columns else 'tozeroy',
            fillcolor='rgba(255, 0, 0, 0.1)'
        ))

        # Ajouter la courbe VOLD en escalier
        fig.add_trace(go.Scatter(
            x=charge_curve['Tranche Horaire'],
            y=charge_curve['VOLD (HD)'],
            mode='lines',
            line=dict(shape='hv', color='#87CEEB', width=3),  # hv = horizontal puis vertical
            name='VOLD (Départs)',
            fill='tozeroy',
            fillcolor='rgba(135, 206, 235, 0.1)'
        ))

        fig.update_layout(
            title="Courbe de Charge par Tranche de 15 Minutes (Escalier)",
            xaxis=dict(
                title="Heure",
                tickformat="%H:%M",
                tickangle=45,
                gridcolor='rgba(128, 128, 128, 0.2)'
            ),
            yaxis=dict(
                title="Nombre de Vols",
                gridcolor='rgba(128, 128, 128, 0.2)'
            ),
            legend_title_text="Type de Vol",
            template="plotly_white",
            width=1500,
            height=600,
            plot_bgcolor='rgba(240, 248, 255, 0.8)',
            hovermode='x unified'
        )

        return fig

    except Exception as e:
        st.error(f"Erreur lors de la création de la courbe de charge : {e}")
        return None


def create_pdf_report(data, selected_date, gantt_chart):
    """Crée un rapport PDF"""
    temp_dir = tempfile.gettempdir()
    pdf_path = os.path.join(temp_dir, f"rapport_vols_{selected_date.strftime('%Y%m%d')}.pdf")

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=landscape(A4),
        rightMargin=30,
        leftMargin=30,
        topMargin=30,
        bottomMargin=30
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        spaceAfter=30,
        alignment=1
    )

    elements = []
    elements.append(Paragraph(f"Rapport des vols - {selected_date.strftime('%d/%m/%Y')}", title_style))
    elements.append(Spacer(1, 20))

    # Ajouter la synthèse RH
    hours_summary = calculate_hours_and_etp_summary(data, selected_date, 8)
    if hours_summary and 'TOTAL' in hours_summary:
        total_data = hours_summary['TOTAL']
        elements.append(Paragraph("Synthèse des ressources humaines", styles['Heading2']))
        elements.append(Paragraph(f"Total vacations: {total_data['nb_vacations']}", styles['Normal']))
        elements.append(Paragraph(f"Total vols: {total_data['nb_vols']}", styles['Normal']))
        elements.append(Paragraph(f"Total heures: {total_data['heures_totales']}h", styles['Normal']))
        elements.append(Paragraph(f"Total ETP: {total_data['etp']}", styles['Normal']))
        elements.append(Spacer(1, 20))

    # Sauvegarder temporairement le graphique si fourni
    if gantt_chart is not None:
        temp_gantt_path = os.path.join(temp_dir, "temp_gantt.png")
        gantt_chart.write_image(temp_gantt_path, width=1000, height=600)
        elements.append(Image(temp_gantt_path, width=700, height=420))

    doc.build(elements)

    with open(pdf_path, 'rb') as pdf_file:
        pdf_data = pdf_file.read()

    # Nettoyer les fichiers temporaires
    os.remove(pdf_path)
    if gantt_chart is not None:
        os.remove(temp_gantt_path)

    return pdf_data


def export_gantt_to_pdf(fig):
    """Exporte le Gantt en PDF"""
    export_fig = fig
    export_fig.update_layout(
        paper_bgcolor='white',
        plot_bgcolor='rgba(240, 248, 255, 0.8)',
        width=1700,
        height=800
    )

    buffer = io.BytesIO()
    export_fig.write_image(
        buffer,
        format="pdf",
        engine="kaleido",
        scale=2
    )

    pdf_data = buffer.getvalue()
    buffer.close()
    return pdf_data


def main():
    st.markdown('<h1 class="main-header">✈️ Système de Planification Ramp Handling</h1>', unsafe_allow_html=True)

    # Initialisation du système
    if 'planning_system' not in st.session_state:
        st.session_state.planning_system = RampPlanningSystem()

    system = st.session_state.planning_system

    # Sidebar pour le chargement de fichier
    with st.sidebar:
        st.header("📁 Chargement des données")
        data_file = st.file_uploader("Charger un fichier Excel", type=['xlsx'])

    if data_file is not None:
        try:
            # Charger les données avec pd.ExcelFile
            uploaded_data = pd.ExcelFile(data_file)
            sheet = uploaded_data.sheet_names[0]  # Charger la première feuille
            raw_data = uploaded_data.parse(sheet)

            # Préprocesser les données
            processed_data = cached_preprocess_data(raw_data)

            # Sidebar pour sélection de date
            with st.sidebar:
                st.header("🔧 Paramètres")
                available_dates = pd.to_datetime(processed_data['DATE'].unique())
                selected_date = st.selectbox(
                    "Sélectionner une date",
                    available_dates,
                    format_func=lambda x: x.strftime('%d/%m/%Y')
                )

                # Options d'affichage
                show_flight_list = st.checkbox("Afficher la liste des vols")
                show_charge_curve = st.checkbox("Afficher la courbe de charge", True)
                show_gantt = st.checkbox("Afficher le diagramme de Gantt", True)

            # Filtrer les données pour la date sélectionnée
            filtered_data = processed_data[pd.to_datetime(processed_data['DATE']) == selected_date]

            # Tabs principaux
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📊 Vue d'ensemble",
                "📅 Planning Gantt",
                "📈 Courbe de charge",
                "🎯 Besoins équipes",
                "💼 Synthèse RH"
            ])

            with tab1:
                st.markdown('<h2 class="section-header">Vue d\'ensemble</h2>', unsafe_allow_html=True)

                # Métriques générales
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total vols", len(filtered_data))
                with col2:
                    total_pax = filtered_data['PAX'].sum() if 'PAX' in filtered_data.columns else 0
                    st.metric("Total passagers", f"{total_pax:,}")
                with col3:
                    companies = filtered_data['Company'].nunique()
                    st.metric("Compagnies", companies)
                with col4:
                    zones = filtered_data['ZONE'].nunique()
                    st.metric("Zones actives", zones)

                # Répartition par zone
                if len(filtered_data) > 0:
                    col1, col2 = st.columns(2)
                    with col1:
                        zone_stats = filtered_data['ZONE'].value_counts()
                        fig_pie = px.pie(values=zone_stats.values, names=zone_stats.index,
                                         title="Répartition des vols par zone")
                        st.plotly_chart(fig_pie, use_container_width=True)

                    with col2:
                        company_stats = filtered_data['Company'].value_counts().head(10)
                        fig_bar = px.bar(x=company_stats.values, y=company_stats.index,
                                         orientation='h', title="Top 10 compagnies")
                        st.plotly_chart(fig_bar, use_container_width=True)

                # Affichage optionnel de la liste des vols
                if show_flight_list:
                    st.subheader("Liste des vols")
                    st.dataframe(filtered_data, use_container_width=True)

            with tab2:
                st.markdown('<h2 class="section-header">Planning Gantt</h2>', unsafe_allow_html=True)

                if show_gantt and len(filtered_data) > 0:
                    gantt_result = create_interactive_gantt(filtered_data, selected_date, system)

                    if gantt_result[0] is not None:  # Si le graphique a pu être créé
                        gantt_chart, processed_gantt_data, total_vacations = gantt_result
                        st.plotly_chart(gantt_chart, use_container_width=True)

                        # Boutons d'export
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("📄 Exporter rapport PDF"):
                                try:
                                    pdf_data = create_pdf_report(filtered_data, selected_date, gantt_chart)
                                    st.download_button(
                                        label="📥 Télécharger rapport",
                                        data=pdf_data,
                                        file_name=f"rapport_vols_{selected_date.strftime('%Y%m%d')}.pdf",
                                        mime="application/pdf"
                                    )
                                except Exception as e:
                                    st.error(f"Erreur lors de la création du PDF : {str(e)}")

                        with col2:
                            if st.button("📊 Exporter Gantt PDF"):
                                try:
                                    pdf_data = export_gantt_to_pdf(gantt_chart)
                                    st.download_button(
                                        label="📥 Télécharger Gantt",
                                        data=pdf_data,
                                        file_name=f"planning_vols_{selected_date.strftime('%d_%m_%Y')}.pdf",
                                        mime="application/pdf"
                                    )
                                except Exception as e:
                                    st.error(f"Erreur lors de l'export : {str(e)}")
                                    st.info("Assurez-vous d'avoir installé le package kaleido : pip install kaleido")
                    else:
                        st.warning("Impossible de créer le diagramme de Gantt pour cette date")
                else:
                    st.info("Aucune donnée à afficher pour cette date")

            with tab3:
                st.markdown('<h2 class="section-header">Courbe de charge</h2>', unsafe_allow_html=True)

                if show_charge_curve and len(filtered_data) > 0:
                    charge_fig = create_charge_curve(filtered_data, selected_date)
                    if charge_fig:
                        st.plotly_chart(charge_fig, use_container_width=True)
                    else:
                        st.info("Pas de données pour créer la courbe de charge")
                else:
                    st.info("Aucune donnée à afficher pour cette date")

            with tab5:
                st.markdown('<h2 class="section-header">Synthèse des ressources humaines</h2>', unsafe_allow_html=True)

                if len(filtered_data) > 0:
                    # Paramètres pour la synthèse
                    col1, col2 = st.columns(2)
                    with col1:
                        vacation_amplitude_summary = st.selectbox(
                            "Amplitude des vacations (heures)",
                            options=[4, 6, 8, 10, 12],
                            index=2,  # 8 heures par défaut
                            key="vacation_amplitude_summary"
                        )
                    with col2:
                        st.write("")

                    # Calculer la synthèse
                    hours_summary = calculate_hours_and_etp_summary(
                        processed_data, selected_date, vacation_amplitude_summary
                    )

                    if hours_summary:
                        st.subheader("📊 Synthèse par zone")

                        # Créer un DataFrame pour l'affichage
                        summary_data = []
                        for zone, data in hours_summary.items():
                            if zone != 'TOTAL':
                                summary_data.append({
                                    'Zone': zone,
                                    'Nb Vacations': data['nb_vacations'],
                                    'Nb Vols': data['nb_vols'],
                                    'Heures Totales': f"{data['heures_totales']}h",
                                    'ETP': data['etp']
                                })

                        df_summary = pd.DataFrame(summary_data)
                        st.dataframe(df_summary, use_container_width=True)

                        # Affichage des totaux
                        st.subheader("🎯 Totaux généraux")
                        total_data = hours_summary.get('TOTAL', {})

                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric(
                                "Total Vacations",
                                total_data.get('nb_vacations', 0)
                            )
                        with col2:
                            st.metric(
                                "Total Vols",
                                total_data.get('nb_vols', 0)
                            )
                        with col3:
                            st.metric(
                                "Total Heures",
                                f"{total_data.get('heures_totales', 0)}h"
                            )
                        with col4:
                            st.metric(
                                "Total ETP",
                                total_data.get('etp', 0)
                            )

                        # Graphique de répartition des ETP par zone
                        if len(summary_data) > 0:
                            col1, col2 = st.columns(2)

                            with col1:
                                # Graphique en secteurs des ETP
                                zones = [d['Zone'] for d in summary_data]
                                etps = [hours_summary[zone]['etp'] for zone in zones]

                                fig_pie = px.pie(
                                    values=etps,
                                    names=zones,
                                    title="Répartition des ETP par zone"
                                )
                                st.plotly_chart(fig_pie, use_container_width=True)

                            with col2:
                                # Graphique en barres des heures
                                heures = [hours_summary[zone]['heures_totales'] for zone in zones]

                                fig_bar = px.bar(
                                    x=zones,
                                    y=heures,
                                    title="Heures totales par zone",
                                    labels={'x': 'Zone', 'y': 'Heures'}
                                )
                                st.plotly_chart(fig_bar, use_container_width=True)

                        # Informations complémentaires
                        st.subheader("ℹ️ Informations")
                        st.info(
                            f"**Principe de calcul :**\n"
                            f"• Une équipe par zone par tranche de {vacation_amplitude_summary}h\n"
                            f"• Chaque équipe peut traiter plusieurs vols simultanément\n"
                            f"• ETP = Heures totales ÷ 8h (journée standard)\n"
                            f"• Équipe = 4 agents (CZ + MOP + PUSH + AGENT)"
                        )
                    else:
                        st.info("Aucune donnée pour calculer la synthèse RH")
                else:
                    st.info("Aucune donnée à analyser pour cette date")
            with tab4:
                st.markdown('<h2 class="section-header">Besoins en équipes</h2>', unsafe_allow_html=True)

                if len(filtered_data) > 0:
                    requirements = calculate_team_requirements_from_flights(
                        processed_data, selected_date, system
                    )

                    if requirements:
                        for shift_name, shift_req in requirements.items():
                            st.subheader(f"Vacation {shift_name}")

                            col1, col2, col3 = st.columns(3)
                            total_teams = sum(zone_req["teams"] for zone_req in shift_req.values())
                            total_flights = sum(zone_req["flights_count"] for zone_req in shift_req.values())

                            with col1:
                                st.metric("Équipes nécessaires", total_teams)
                            with col2:
                                st.metric("Vols à traiter", total_flights)
                            with col3:
                                total_agents = total_teams * 4  # 4 qualifications par équipe
                                st.metric("Agents nécessaires", total_agents)

                            # Détail par zone
                            st.write("**Détail par zone (1 équipe par zone max):**")
                            zone_data = []
                            for zone, zone_req in shift_req.items():
                                if zone_req["teams"] > 0:
                                    zone_data.append({
                                        "Zone": zone,
                                        "Équipe": "✅ 1 équipe" if zone_req["teams"] > 0 else "❌ Aucune",
                                        "Vols simultanés": zone_req["flights_count"],
                                        "CZ": zone_req["qualifications"]["CZ"],
                                        "MOP": zone_req["qualifications"]["MOP"],
                                        "PUSH": zone_req["qualifications"]["PUSH"],
                                        "AGENT": zone_req["qualifications"]["AGENT"]
                                    })

                            if zone_data:
                                df_zones = pd.DataFrame(zone_data)
                                st.dataframe(df_zones, use_container_width=True)

                                st.info(
                                    "💡 **Principe :** Une seule équipe par zone peut traiter "
                                    "plusieurs vols simultanément selon les besoins."
                                )
                            else:
                                st.info(f"Aucun vol prévu pour la vacation {shift_name}")
                    else:
                        st.info("Aucun besoin calculé pour cette date")
                else:
                    st.info("Aucune donnée à analyser pour cette date")

        except Exception as e:
            st.error(f"Erreur lors du traitement du fichier : {str(e)}")
            st.info("Vérifiez que votre fichier Excel contient les colonnes : Date, n°arr, n°dep, Arr, Dép, Parking")

            # Afficher les colonnes disponibles pour diagnostic
            try:
                uploaded_data = pd.ExcelFile(data_file)
                sheet = uploaded_data.sheet_names[0]
                raw_data = uploaded_data.parse(sheet)
                st.write("**Colonnes disponibles dans le fichier :**")
                st.write(list(raw_data.columns))
            except:
                pass

    else:
        st.info("👆 Veuillez charger un fichier Excel pour commencer l'analyse")

        # Afficher les informations sur le format attendu
        st.subheader("📋 Format de fichier attendu")
        st.write("Votre fichier Excel doit contenir les colonnes suivantes :")

        col1, col2 = st.columns(2)
        with col1:
            st.write("**Colonnes obligatoires :**")
            st.write("• `Date` - Date du vol")
            st.write("• `n°arr` - Numéro de vol arrivée")
            st.write("• `n°dep` - Numéro de vol départ")
            st.write("• `Arr` - Heure d'arrivée (format: 0620 pour 06:20)")
            st.write("• `Dép` - Heure de départ (format: 0620 pour 06:20)")

        with col2:
            st.write("**Colonnes facultatives :**")
            st.write("• `Parking` - Numéro du parking")
            st.write("• `SO` - Nombre de passagers")
            st.write("• `Origine` - Aéroport d'origine")
            st.write("• `Destination` - Aéroport de destination")

        # Afficher la structure des zones attendue
        st.subheader("📍 Structure des zones")
        col1, col2 = st.columns(2)

        with col1:
            for i, (zone, parkings) in enumerate(list(system.zones.items())[:3]):
                st.write(f"**{zone}:** {', '.join(parkings)}")

        with col2:
            for i, (zone, parkings) in enumerate(list(system.zones.items())[3:]):
                st.write(f"**{zone}:** {', '.join(parkings)}")


if __name__ == "__main__":
    main()
