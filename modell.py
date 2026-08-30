#!/usr/bin/env python3
"""
fussball_modell.py — Dixon-Coles Wahrscheinlichkeitsmodell fuer Fussballspiele

Was es macht:
  1. Laedt historische Spieldaten (football-data.co.uk, kostenlos, inkl. Quoten)
  2. Schaetzt fuer jedes Team eine Angriffs- und eine Abwehrstaerke
     (Maximum-Likelihood, mit Zeitgewichtung: alte Spiele zaehlen weniger)
  3. Berechnet daraus die Wahrscheinlichkeitsverteilung ueber alle Ergebnisse
  4. Vergleicht mit den Buchmacherquoten und zeigt, wo ein Value liegt
  5. Backtest: prueft ueber vergangene Saisons, ob das Modell wirklich Geld
     gemacht haette (Walk-Forward, kein Blick in die Zukunft)

Benutzung:
  python3 fussball_modell.py --liga D1 --saisons 2122 2223 2324 2425 2526
  python3 fussball_modell.py --liga D1 --backtest
  python3 fussball_modell.py --liga D1 --prognose "Bayern Munich" "Dortmund"

Liga-Codes: D1=Bundesliga, D2=2.Bundesliga, E0=Premier League,
            SP1=La Liga, I1=Serie A, F1=Ligue 1, N1=Eredivisie, ...
"""

import argparse
import json
import os
import sys
from datetime import timedelta

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson


# Alle Ligen, die football-data.co.uk fuehrt
LIGEN = {
    "D1":  "Bundesliga",              "D2":  "2. Bundesliga",
    "E0":  "Premier League",          "E1":  "Championship",
    "E2":  "League One",              "E3":  "League Two",
    "EC":  "National League",
    "SP1": "La Liga",                 "SP2": "Segunda Division",
    "I1":  "Serie A",                 "I2":  "Serie B",
    "F1":  "Ligue 1",                 "F2":  "Ligue 2",
    "N1":  "Eredivisie",              "B1":  "Belgien Pro League",
    "P1":  "Portugal Primeira",       "T1":  "Tuerkei Sueper Lig",
    "SC0": "Schottland Premiership",  "SC1": "Schottland Championship",
    "G1":  "Griechenland Super League",
}

BASE_URL = "https://www.football-data.co.uk/mmz4281/{saison}/{liga}.csv"
CACHE = "daten"
MAX_TORE = 10

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="theme-color" content="#0038A8">
<meta name="mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="Wettrechner">
<title>Wettrechner — Dixon-Coles</title>
<style>
  :root{
    --ink:#15171B; --paper:#EFEDE7; --board:#0F2E2C; --board-2:#17403D;
    --signal:#2F9E68; --clay:#BE4429; --sand:#DED8C6; --muted:#6B6A63;
    --mono: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
  }
  *{box-sizing:border-box}
  body{
    margin:0; background:var(--paper); color:var(--ink);
    font-family:-apple-system,"Helvetica Neue",Arial,sans-serif;
    font-size:15px; line-height:1.55;
  }
  .wrap{max-width:940px; margin:0 auto; padding:28px 20px 64px}

  header{border-bottom:2px solid var(--ink); padding-bottom:12px; margin-bottom:26px}
  h1{
    font-size:19px; font-weight:700; margin:0; letter-spacing:.16em;
    text-transform:uppercase;
  }
  header p{margin:5px 0 0; font-size:13px; color:var(--muted)}

  .panel{background:#fff; border:1px solid var(--sand); padding:20px; margin-bottom:20px}
  .panel h2{
    font-size:11px; letter-spacing:.14em; text-transform:uppercase;
    margin:0 0 14px; color:var(--muted); font-weight:700;
  }

  .grid{display:grid; grid-template-columns:1fr auto 1fr; gap:14px; align-items:end}
  .vs{font-family:var(--mono); font-size:13px; color:var(--muted); padding-bottom:11px}
  label{display:block; font-size:11px; letter-spacing:.1em; text-transform:uppercase;
        color:var(--muted); margin-bottom:5px; font-weight:700}
  select,input{
    width:100%; padding:9px 10px; border:1px solid #C9C4B4; background:#fff;
    font-size:15px; font-family:inherit; color:var(--ink); border-radius:0;
  }
  input{font-family:var(--mono)}
  select:focus,input:focus{outline:2px solid var(--board); outline-offset:1px}

  .quoten{display:grid; grid-template-columns:repeat(3,1fr); gap:14px; margin-top:18px}
  .hint{font-size:12px; color:var(--muted); margin:10px 0 0}

  button{
    margin-top:18px; padding:12px 26px; border:0; background:var(--ink); color:#fff;
    font-size:12px; letter-spacing:.14em; text-transform:uppercase; font-weight:700;
    cursor:pointer; font-family:inherit;
  }
  button:hover{background:var(--board)}
  button:focus-visible{outline:3px solid var(--signal); outline-offset:2px}
  button.ghost{background:none; color:var(--muted); border:1px solid #C9C4B4; margin-top:0;
               padding:7px 14px; letter-spacing:.08em}
  button.ghost:hover{background:var(--sand); color:var(--ink)}

  /* Signaturelement: Anzeigetafel */
  .board{background:var(--board); color:#F2F5F1; padding:22px; margin-bottom:20px}
  .board .teams{
    font-size:12px; letter-spacing:.14em; text-transform:uppercase;
    color:#8FB3AC; margin-bottom:16px;
  }
  .bar{display:flex; height:52px; overflow:hidden; margin-bottom:6px}
  .seg{
    display:flex; flex-direction:column; align-items:center; justify-content:center;
    font-family:var(--mono); color:#0F2E2C; transition:flex-basis .45s ease;
    min-width:0; overflow:hidden;
  }
  .seg b{font-size:19px; font-weight:700; line-height:1.1}
  .seg span{font-size:10px; letter-spacing:.16em; opacity:.72}
  .s1{background:#7FD0A6} .sx{background:#BFCBC2} .s2{background:#E3B25C}

  .xg{font-family:var(--mono); font-size:13px; color:#8FB3AC; margin-top:14px}
  .xg b{color:#F2F5F1; font-weight:600}

  table{width:100%; border-collapse:collapse; font-size:14px}
  th{
    text-align:left; font-size:10px; letter-spacing:.12em; text-transform:uppercase;
    color:var(--muted); border-bottom:1px solid var(--sand); padding:0 8px 7px; font-weight:700;
  }
  th.num,td.num{text-align:right; font-family:var(--mono)}
  td{padding:9px 8px; border-bottom:1px solid #F0EEE7}
  tr:last-child td{border-bottom:0}
  .pos{color:var(--signal); font-weight:600}
  .neg{color:var(--clay)}
  .tag{
    display:inline-block; font-size:10px; letter-spacing:.1em; text-transform:uppercase;
    padding:2px 7px; font-weight:700;
  }
  .tag.value{background:#DFF2E7; color:#1B5C3C}
  .tag.warn{background:#FBE7D8; color:#8A3418}

  .matrix{border-collapse:collapse; font-family:var(--mono); font-size:12px; width:auto}
  .matrix td,.matrix th{border:1px solid #fff; padding:0; text-align:center}
  .matrix .cell{width:46px; height:32px; line-height:32px; color:#15171B}
  .matrix .ax{color:var(--muted); font-size:10px; width:46px; height:22px; line-height:22px;
              border:0; letter-spacing:.08em}
  .mrow{display:flex; gap:26px; flex-wrap:wrap; align-items:flex-start}
  .legend{font-size:12px; color:var(--muted); max-width:260px}

  .split{display:grid; grid-template-columns:1fr 1fr; gap:20px}
  .note{
    border-left:3px solid var(--clay); padding:2px 0 2px 14px; font-size:13.5px;
    color:#4A4842; margin-top:16px;
  }
  textarea{
    width:100%; height:150px; font-family:var(--mono); font-size:12px; padding:10px;
    border:1px solid #C9C4B4; resize:vertical;
  }
  .status{font-size:13px; margin-top:10px; font-family:var(--mono)}
  details summary{cursor:pointer; font-size:12px; letter-spacing:.1em;
                  text-transform:uppercase; color:var(--muted); font-weight:700}
  #ausgabe[hidden]{display:none}
  .err{color:var(--clay); font-size:13px; margin-top:10px}
  @media (max-width:760px){
    .wrap{padding:18px 14px 48px}
    .grid{grid-template-columns:1fr; gap:12px} .vs{display:none}
    .split{grid-template-columns:1fr}
    .quoten{grid-template-columns:repeat(3,1fr); gap:8px}
    .panel{padding:16px}
    select,input{font-size:16px; padding:13px 11px}
    button{width:100%; padding:16px; font-size:13px}
    .board{padding:18px 16px}
    .bar{height:64px}
    .seg b{font-size:17px}
    .mrow{display:block}
    .matrix{width:100%}
    .matrix .cell,.matrix .ax{width:auto; font-size:11px}
    .matrix .cell{height:38px; line-height:38px}
    .legend{max-width:none; margin-top:14px}
    table{font-size:15px}
    td{padding:11px 6px}
    textarea{font-size:14px}
  }
  @media (max-width:380px){
    .matrix .cell{font-size:10px}
  }
</style>
</head>
<body>
<div class="wrap">

<header>
  <h1>Wettrechner</h1>
  <p>Dixon-Coles Modell &middot; <span id="standinfo">europäische Ligen</span></p>
</header>

<div class="panel">
  <h2>Spiel</h2>
  <div style="margin-bottom:16px">
    <label for="liga">Liga</label>
    <select id="liga"></select>
  </div>
  <div class="grid">
    <div>
      <label for="heim">Heimmannschaft</label>
      <select id="heim"></select>
    </div>
    <div class="vs">gegen</div>
    <div>
      <label for="ausw">Auswärtsmannschaft</label>
      <select id="ausw"></select>
    </div>
  </div>

  <div class="quoten">
    <div><label for="q1">Quote 1</label><input id="q1" type="text" inputmode="decimal" placeholder="1.95"></div>
    <div><label for="qx">Quote X</label><input id="qx" type="text" inputmode="decimal" placeholder="3.70"></div>
    <div><label for="q2">Quote 2</label><input id="q2" type="text" inputmode="decimal" placeholder="3.80"></div>
  </div>
  <p class="hint">Quoten sind optional. Ohne sie bekommst du nur die Wahrscheinlichkeiten, mit ihnen zusätzlich den Value-Vergleich.</p>
  <p class="err" id="fehler" hidden></p>
  <button id="rechnen">Berechnen</button>
</div>

<div id="ausgabe" hidden>

  <div class="board">
    <div class="teams" id="paarung"></div>
    <div class="bar">
      <div class="seg s1" id="seg1"><b></b><span>Heim</span></div>
      <div class="seg sx" id="segx"><b></b><span>Remis</span></div>
      <div class="seg s2" id="seg2"><b></b><span>Auswärts</span></div>
    </div>
    <div class="xg" id="xg"></div>
  </div>

  <div class="split">
    <div class="panel">
      <h2>Wahrscheinlichkeiten</h2>
      <table>
        <thead><tr><th>Markt</th><th class="num">Chance</th><th class="num">Faire Quote</th></tr></thead>
        <tbody id="tabelle"></tbody>
      </table>
    </div>

    <div class="panel">
      <h2>Value gegen deine Quote</h2>
      <table>
        <thead><tr><th>Tipp</th><th class="num">Quote</th><th class="num">Edge</th><th class="num">Einsatz</th></tr></thead>
        <tbody id="value"></tbody>
      </table>
      <div id="urteil"></div>
    </div>
  </div>

  <div class="panel">
    <h2>Ergebnismatrix</h2>
    <div class="mrow">
      <table class="matrix" id="matrix"></table>
      <p class="legend">Jede Zelle ist die Wahrscheinlichkeit für genau dieses Ergebnis. Zeilen sind Heimtore, Spalten Auswärtstore. Je dunkler, desto wahrscheinlicher.</p>
    </div>
  </div>
</div>

<div class="panel">
  <h2>Daten</h2>
  <p class="hint" style="margin:0">Die Werte werden jeden Montagmorgen automatisch neu
  berechnet. Der Stand der gewählten Liga steht oben unter dem Titel.</p>
  <details style="margin-top:14px">
    <summary>Werte von Hand ersetzen</summary>
    <p class="hint">Nur nötig, wenn du eigene Zahlen einspielen willst. Im Terminal
    <code>--teams</code> laufen lassen und die Tabelle hier einfügen, inklusive der
    Zeile mit Heimvorteil und rho. Sie ersetzt die oben gewählte Liga.</p>
    <textarea id="paste" placeholder="Bayern Munich              +0.831   -0.097"></textarea>
    <div style="margin-top:10px"><button class="ghost" id="uebernehmen">Werte übernehmen</button></div>
    <p class="status" id="status"></p>
  </details>
</div>

</div>

<script>
var LIGEN = /*__DATEN__*/{};
var MODELL = null;

var MAXT = 10;

function pois(k, lam){
  var f = 1;
  for (var i = 2; i <= k; i++) f *= i;
  return Math.exp(-lam) * Math.pow(lam, k) / f;
}

function matrix(heim, ausw){
  var h = MODELL.teams[heim], a = MODELL.teams[ausw];
  var lam = Math.exp(h[0] + a[1] + MODELL.gamma);
  var mu  = Math.exp(a[0] + h[1]);
  var ph = [], pa = [], i, j;
  for (i = 0; i <= MAXT; i++){ ph.push(pois(i, lam)); pa.push(pois(i, mu)); }
  var m = [], sum = 0;
  for (i = 0; i <= MAXT; i++){
    m.push([]);
    for (j = 0; j <= MAXT; j++) m[i].push(ph[i] * pa[j]);
  }
  var r = MODELL.rho;
  m[0][0] *= 1 - lam * mu * r;
  m[0][1] *= 1 + lam * r;
  m[1][0] *= 1 + mu * r;
  m[1][1] *= 1 - r;
  for (i = 0; i <= MAXT; i++) for (j = 0; j <= MAXT; j++) sum += m[i][j];
  for (i = 0; i <= MAXT; i++) for (j = 0; j <= MAXT; j++) m[i][j] /= sum;
  return {m: m, lam: lam, mu: mu};
}

function auswerten(heim, ausw){
  var r = matrix(heim, ausw), m = r.m;
  var p1 = 0, px = 0, p2 = 0, ueber = 0, btts = 0, i, j;
  for (i = 0; i <= MAXT; i++){
    for (j = 0; j <= MAXT; j++){
      var p = m[i][j];
      if (i > j) p1 += p; else if (i === j) px += p; else p2 += p;
      if (i + j > 2) ueber += p;
      if (i > 0 && j > 0) btts += p;
    }
  }
  return {m: m, lam: r.lam, mu: r.mu, p1: p1, px: px, p2: p2,
          ueber: ueber, btts: btts};
}

function kelly(p, q){
  var b = q - 1;
  if (b <= 0) return 0;
  return Math.max(0, (p * b - (1 - p)) / b) * 0.25;
}

function pct(x){ return (x * 100).toFixed(1) + " %"; }

function fuelleLigen(){
  var sel = document.getElementById("liga");
  var codes = Object.keys(LIGEN).sort(function(a,b){
    return LIGEN[a].name.localeCompare(LIGEN[b].name);
  });
  sel.innerHTML = "";
  codes.forEach(function(c){
    var o = document.createElement("option");
    o.value = c;
    o.textContent = LIGEN[c].name + "  (" + Object.keys(LIGEN[c].teams).length + " Teams)";
    sel.appendChild(o);
  });
  ligaWechsel();
}

function ligaWechsel(){
  var code = document.getElementById("liga").value;
  MODELL = LIGEN[code];
  document.getElementById("standinfo").textContent =
    MODELL.name + " \u00b7 Stand " + (MODELL.stand || "unbekannt");
  document.getElementById("ausgabe").hidden = true;
  fuelleTeams();
}

function fuelleTeams(){
  var namen = Object.keys(MODELL.teams).sort();
  ["heim","ausw"].forEach(function(id){
    var sel = document.getElementById(id), alt = sel.value;
    sel.innerHTML = "";
    namen.forEach(function(n){
      var o = document.createElement("option");
      o.value = n; o.textContent = n;
      sel.appendChild(o);
    });
    if (namen.indexOf(alt) >= 0) sel.value = alt;
  });
  if (namen.length > 1) document.getElementById("ausw").selectedIndex = 1;
}

function quote(id){
  var v = document.getElementById(id).value.trim().replace(",", ".");
  if (v === "") return null;
  var n = parseFloat(v);
  return (isFinite(n) && n > 1) ? n : NaN;
}

function rechnen(){
  var heim = document.getElementById("heim").value;
  var ausw = document.getElementById("ausw").value;
  var fehler = document.getElementById("fehler");
  fehler.hidden = true;

  if (heim === ausw){
    fehler.textContent = "Wähle zwei verschiedene Mannschaften.";
    fehler.hidden = false;
    return;
  }
  var qs = [quote("q1"), quote("qx"), quote("q2")];
  for (var k = 0; k < 3; k++){
    if (isNaN(qs[k])){
      fehler.textContent = "Quoten müssen Zahlen über 1.00 sein, zum Beispiel 2.35.";
      fehler.hidden = false;
      return;
    }
  }

  var r = auswerten(heim, ausw);
  document.getElementById("ausgabe").hidden = false;
  document.getElementById("paarung").textContent = heim + "  —  " + ausw;
  document.getElementById("xg").innerHTML =
    "Erwartete Tore <b>" + r.lam.toFixed(2) + " : " + r.mu.toFixed(2) + "</b>";

  var segs = [["seg1", r.p1], ["segx", r.px], ["seg2", r.p2]];
  segs.forEach(function(s){
    var el = document.getElementById(s[0]);
    el.style.flexBasis = (s[1] * 100) + "%";
    el.querySelector("b").textContent = (s[1] * 100).toFixed(0) + "%";
  });

  var zeilen = [
    ["Heimsieg", r.p1], ["Unentschieden", r.px], ["Auswärtssieg", r.p2],
    ["Über 2.5 Tore", r.ueber], ["Unter 2.5 Tore", 1 - r.ueber],
    ["Beide Teams treffen", r.btts]
  ];
  document.getElementById("tabelle").innerHTML = zeilen.map(function(z){
    return "<tr><td>" + z[0] + "</td><td class='num'>" + pct(z[1]) +
           "</td><td class='num'>" + (1 / z[1]).toFixed(2) + "</td></tr>";
  }).join("");

  var namen = ["1  Heim", "X  Remis", "2  Auswärts"];
  var ps = [r.p1, r.px, r.p2];
  var beste = -1, bestEdge = 0, html = "";
  for (var i = 0; i < 3; i++){
    if (qs[i] === null){
      html += "<tr><td>" + namen[i] + "</td><td class='num'>—</td>" +
              "<td class='num'>—</td><td class='num'>—</td></tr>";
      continue;
    }
    var edge = ps[i] * qs[i] - 1;
    var f = kelly(ps[i], qs[i]);
    if (edge > bestEdge){ bestEdge = edge; beste = i; }
    html += "<tr><td>" + namen[i] + "</td>" +
            "<td class='num'>" + qs[i].toFixed(2) + "</td>" +
            "<td class='num " + (edge > 0 ? "pos" : "neg") + "'>" +
            (edge > 0 ? "+" : "") + (edge * 100).toFixed(1) + " %</td>" +
            "<td class='num'>" + (f > 0 ? (f * 100).toFixed(1) + " %" : "—") + "</td></tr>";
  }
  document.getElementById("value").innerHTML = html;

  var u = document.getElementById("urteil");
  if (qs[0] === null && qs[1] === null && qs[2] === null){
    u.innerHTML = "<p class='hint'>Trag Quoten ein, um den Value zu sehen.</p>";
  } else if (beste < 0){
    u.innerHTML = "<p class='note'>Kein Value. Bei diesen Quoten liegt das Modell " +
      "überall unter dem Buchmacher. Das ist der Normalfall — kein Grund, trotzdem zu setzen.</p>";
  } else if (bestEdge > 0.15){
    u.innerHTML = "<p style='margin:14px 0 0'><span class='tag warn'>Prüfen</span></p>" +
      "<p class='note'>Ein Edge von " + (bestEdge * 100).toFixed(0) + " % ist zu groß, " +
      "um echt zu sein. Meist fehlt dem Modell eine Information, die der Buchmacher hat: " +
      "Verletzungen, Trainerwechsel, Transfers, englische Woche. Erst suchen, dann entscheiden.</p>";
  } else if (bestEdge > 0.05){
    u.innerHTML = "<p style='margin:14px 0 0'><span class='tag value'>Value auf " +
      namen[beste].slice(0, 1) + "</span></p>" +
      "<p class='note'>Einsatz ist Viertel-Kelly. Halbier ihn nochmal, wenn du dir " +
      "bei der Datenlage nicht sicher bist.</p>";
  } else {
    u.innerHTML = "<p class='note'>Edge unter 5 %. Zu dünn — das steckt im Schätzrauschen " +
      "des Modells und ist kein belastbarer Vorteil.</p>";
  }

  var maxp = 0, i2, j2;
  for (i2 = 0; i2 <= 5; i2++) for (j2 = 0; j2 <= 5; j2++)
    if (r.m[i2][j2] > maxp) maxp = r.m[i2][j2];
  var t = "<tr><td class='ax'></td>";
  for (j2 = 0; j2 <= 5; j2++) t += "<td class='ax'>" + j2 + "</td>";
  t += "</tr>";
  for (i2 = 0; i2 <= 5; i2++){
    t += "<tr><td class='ax'>" + i2 + "</td>";
    for (j2 = 0; j2 <= 5; j2++){
      var v = r.m[i2][j2], rel = v / maxp;
      var bg = "rgba(15,46,44," + (0.06 + rel * 0.82).toFixed(3) + ")";
      var col = rel > 0.55 ? "#F2F5F1" : "#15171B";
      t += "<td class='cell' style='background:" + bg + ";color:" + col + "'>" +
           (v * 100).toFixed(1) + "</td>";
    }
    t += "</tr>";
  }
  document.getElementById("matrix").innerHTML = t;
}

function uebernehmen(){
  var txt = document.getElementById("paste").value;
  var status = document.getElementById("status");
  var zeilen = txt.split("\n");
  var neu = {}, gamma = null, rho = null, n = 0;

  for (var i = 0; i < zeilen.length; i++){
    var z = zeilen[i];
    var hv = z.match(/Heimvorteil\s*([+-]?[\d.]+)\s*,\s*rho\s*([+-]?[\d.]+)/i);
    if (hv){ gamma = parseFloat(hv[1]); rho = parseFloat(hv[2]); continue; }
    var m = z.match(/^(.+?)\s{2,}([+-]\d*\.\d+)\s+([+-]\d*\.\d+)\s*$/);
    if (m){
      var name = m[1].trim();
      if (name.toLowerCase() === "team") continue;
      neu[name] = [parseFloat(m[2]), parseFloat(m[3])];
      n++;
    }
  }

  if (n < 2){
    status.style.color = "#BE4429";
    status.textContent = "Keine Teamwerte erkannt. Kopier die ganze Tabelle inklusive der Zahlenspalten.";
    return;
  }
  MODELL.teams = neu;
  if (gamma !== null){ MODELL.gamma = gamma; MODELL.rho = rho; }
  MODELL.stand = "manuell";
  document.getElementById("standinfo").textContent = MODELL.name + " \u00b7 Stand manuell";
  fuelleTeams();
  document.getElementById("ausgabe").hidden = true;
  status.style.color = "#2F9E68";
  status.textContent = n + " Teams übernommen" +
    (gamma !== null ? ", Heimvorteil " + gamma.toFixed(3) + ", rho " + rho.toFixed(3) : "") + ".";
}

fuelleLigen();
document.getElementById("liga").addEventListener("change", ligaWechsel);
document.getElementById("rechnen").addEventListener("click", rechnen);
document.getElementById("uebernehmen").addEventListener("click", uebernehmen);
["q1","qx","q2"].forEach(function(id){
  document.getElementById(id).addEventListener("keydown", function(e){
    if (e.key === "Enter") rechnen();
  });
});
</script>
</body>
</html>
"""


# --------------------------------------------------------------------------
# 1. Daten laden
# --------------------------------------------------------------------------

def lade_daten(liga, saisons, cache_dir=CACHE):
    """Laedt CSVs von football-data.co.uk und legt sie lokal ab."""
    os.makedirs(cache_dir, exist_ok=True)
    teile = []
    for s in saisons:
        pfad = os.path.join(cache_dir, f"{liga}_{s}.csv")
        if not os.path.exists(pfad):
            url = BASE_URL.format(saison=s, liga=liga)
            print(f"  lade {url}")
            try:
                pd.read_csv(url, encoding="latin-1").to_csv(pfad, index=False)
            except Exception as e:
                print(f"  !! {s} fehlgeschlagen: {e}")
                continue
        df = pd.read_csv(pfad, encoding="latin-1")
        df["Saison"] = s
        teile.append(df)

    if not teile:
        sys.exit("Keine Daten geladen.")

    df = pd.concat(teile, ignore_index=True)
    return aufbereiten(df)


def aufbereiten(df):
    """Vereinheitlicht Spalten, wirft unvollstaendige Zeilen raus."""
    pflicht = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
    fehlt = [c for c in pflicht if c not in df.columns]
    if fehlt:
        sys.exit(f"Spalten fehlen: {fehlt}")

    df = df.dropna(subset=pflicht).copy()
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, format="mixed")
    df["FTHG"] = df["FTHG"].astype(int)
    df["FTAG"] = df["FTAG"].astype(int)

    # Quoten: PSC/PSH = Pinnacle Schlussquote (Referenz), sonst Bet365
    for ziel, kandidaten in [("QH", ["PSCH", "PSH", "B365H", "AvgH", "BbAvH"]),
                             ("QD", ["PSCD", "PSD", "B365D", "AvgD", "BbAvD"]),
                             ("QA", ["PSCA", "PSA", "B365A", "AvgA", "BbAvA"])]:
        df[ziel] = np.nan
        for k in kandidaten:
            if k in df.columns:
                df[ziel] = df[ziel].fillna(pd.to_numeric(df[k], errors="coerce"))

    return df.sort_values("Date").reset_index(drop=True)


# --------------------------------------------------------------------------
# 2. Dixon-Coles Modell
# --------------------------------------------------------------------------

def tau(h, a, lam, mu, rho):
    """
    Dixon-Coles Korrektur. Reines Poisson unterschaetzt 0:0 und 1:1 und
    ueberschaetzt 1:0 und 0:1 — genau das korrigiert dieser Faktor.
    """
    t = np.ones_like(lam, dtype=float)
    m00 = (h == 0) & (a == 0)
    m01 = (h == 0) & (a == 1)
    m10 = (h == 1) & (a == 0)
    m11 = (h == 1) & (a == 1)
    t[m00] = 1 - lam[m00] * mu[m00] * rho
    t[m01] = 1 + lam[m01] * rho
    t[m10] = 1 + mu[m10] * rho
    t[m11] = 1 - rho
    return t


class DixonColes:
    """
    lambda (Heim-Tore)  = exp(angriff_heim + abwehr_ausw + heimvorteil)
    mu     (Ausw-Tore)  = exp(angriff_ausw + abwehr_heim)

    xi steuert die Zeitgewichtung: Gewicht = exp(-xi * Tage_her).
    xi=0.0018 entspricht etwa einer Halbwertszeit von ~1 Jahr.
    """

    def __init__(self, xi=0.0018):
        self.xi = xi
        self.teams = None
        self.params = None

    def fit(self, df, stichtag=None):
        teams = sorted(set(df["HomeTeam"]) | set(df["AwayTeam"]))
        idx = {t: i for i, t in enumerate(teams)}
        n = len(teams)

        h = df["HomeTeam"].map(idx).to_numpy()
        a = df["AwayTeam"].map(idx).to_numpy()
        hg = df["FTHG"].to_numpy()
        ag = df["FTAG"].to_numpy()

        stichtag = stichtag or df["Date"].max()
        tage = (stichtag - df["Date"]).dt.days.to_numpy().astype(float)
        gew = np.exp(-self.xi * np.maximum(tage, 0))

        # Startwerte: Angriff, Abwehr, Heimvorteil, rho
        x0 = np.concatenate([np.zeros(n), np.zeros(n), [0.25], [-0.05]])

        def negloglik(p):
            att = p[:n]
            deff = p[n:2 * n]
            gamma = p[2 * n]
            rho = p[2 * n + 1]

            att = att - att.mean()          # Identifizierbarkeit

            lam = np.exp(att[h] + deff[a] + gamma)
            mu = np.exp(att[a] + deff[h])
            lam = np.clip(lam, 1e-6, 15)
            mu = np.clip(mu, 1e-6, 15)

            t = tau(hg, ag, lam, mu, rho)
            t = np.clip(t, 1e-10, None)

            ll = (np.log(t)
                  + poisson.logpmf(hg, lam)
                  + poisson.logpmf(ag, mu))
            return -np.sum(gew * ll)

        grenzen = [(-3, 3)] * n + [(-3, 3)] * n + [(-1, 1)] + [(-0.2, 0.2)]
        res = minimize(negloglik, x0, method="L-BFGS-B", bounds=grenzen,
                       options={"maxiter": 300})

        p = res.x
        p[:n] = p[:n] - p[:n].mean()
        self.teams = idx
        self.params = p
        self.n = n
        return self

    def erwartungswerte(self, heim, ausw):
        if heim not in self.teams or ausw not in self.teams:
            return None
        n = self.n
        att, deff = self.params[:n], self.params[n:2 * n]
        gamma, rho = self.params[2 * n], self.params[2 * n + 1]
        i, j = self.teams[heim], self.teams[ausw]
        lam = float(np.exp(att[i] + deff[j] + gamma))
        mu = float(np.exp(att[j] + deff[i]))
        return lam, mu, rho

    def matrix(self, heim, ausw):
        """Wahrscheinlichkeitsmatrix ueber alle Ergebnisse bis MAX_TORE."""
        ew = self.erwartungswerte(heim, ausw)
        if ew is None:
            return None
        lam, mu, rho = ew
        h = np.arange(MAX_TORE + 1)
        m = np.outer(poisson.pmf(h, lam), poisson.pmf(h, mu))
        m[0, 0] *= 1 - lam * mu * rho
        m[0, 1] *= 1 + lam * rho
        m[1, 0] *= 1 + mu * rho
        m[1, 1] *= 1 - rho
        return m / m.sum()

    def prognose(self, heim, ausw):
        m = self.matrix(heim, ausw)
        if m is None:
            return None
        tore = np.arange(MAX_TORE + 1)
        gesamt = tore[:, None] + tore[None, :]
        lam, mu, _ = self.erwartungswerte(heim, ausw)
        return {
            "heimsieg": float(np.tril(m, -1).sum()),
            "unentschieden": float(np.trace(m)),
            "auswaertssieg": float(np.triu(m, 1).sum()),
            "ueber_2_5": float(m[gesamt > 2.5].sum()),
            "unter_2_5": float(m[gesamt <= 2.5].sum()),
            "btts": float(m[1:, 1:].sum()),
            "xg_heim": lam,
            "xg_ausw": mu,
            "top_ergebnisse": sorted(
                [((i, j), float(m[i, j])) for i in range(6) for j in range(6)],
                key=lambda x: -x[1])[:5],
        }


# --------------------------------------------------------------------------
# 3. Value und Kelly
# --------------------------------------------------------------------------

def marge_entfernen(quoten):
    """Buchmachermarge rausrechnen -> faire Wahrscheinlichkeiten des Marktes."""
    roh = np.array([1 / q for q in quoten], dtype=float)
    return roh / roh.sum()


def kelly(p, quote, teil=0.25):
    """Fractional Kelly. Voller Kelly ist fuer echtes Geld zu volatil."""
    b = quote - 1
    f = (p * b - (1 - p)) / b
    return max(0.0, f * teil)


# --------------------------------------------------------------------------
# 4. Backtest (Walk-Forward)
# --------------------------------------------------------------------------

def backtest(df, xi=0.0018, min_spiele=380, refit_tage=7,
             min_edge=0.05, einsatz_modus="kelly", bankroll=1000.0):
    """
    Geht die Historie chronologisch durch. Vor jedem Spieltag wird das
    Modell NUR mit Daten trainiert, die bis dahin vorlagen.
    """
    df = df.dropna(subset=["QH", "QD", "QA"]).reset_index(drop=True)
    if len(df) < min_spiele + 50:
        sys.exit("Zu wenig Daten fuer einen Backtest.")

    start = df["Date"].iloc[min_spiele]
    test = df[df["Date"] >= start].copy()

    modell = None
    letztes_fit = None
    zeilen = []

    for _, spiel in test.iterrows():
        datum = spiel["Date"]
        if letztes_fit is None or (datum - letztes_fit) >= timedelta(days=refit_tage):
            train = df[df["Date"] < datum]
            if len(train) < min_spiele:
                continue
            modell = DixonColes(xi=xi).fit(train, stichtag=datum)
            letztes_fit = datum

        pr = modell.prognose(spiel["HomeTeam"], spiel["AwayTeam"])
        if pr is None:
            continue

        quoten = [spiel["QH"], spiel["QD"], spiel["QA"]]
        modell_p = [pr["heimsieg"], pr["unentschieden"], pr["auswaertssieg"]]
        markt_p = marge_entfernen(quoten)

        ergebnis = (0 if spiel["FTHG"] > spiel["FTAG"]
                    else 1 if spiel["FTHG"] == spiel["FTAG"] else 2)

        zeilen.append({
            "Date": datum,
            "Heim": spiel["HomeTeam"], "Ausw": spiel["AwayTeam"],
            "p_modell": modell_p, "p_markt": list(markt_p),
            "quoten": quoten, "ergebnis": ergebnis,
        })

    if not zeilen:
        sys.exit("Backtest lieferte keine Zeilen.")

    res = pd.DataFrame(zeilen)

    # --- Kalibrierung: schlaegt das Modell den Markt? ---
    def logloss(spalte):
        return -np.mean([np.log(max(r[spalte][r["ergebnis"]], 1e-12))
                         for _, r in res.iterrows()])

    ll_modell = logloss("p_modell")
    ll_markt = logloss("p_markt")

    # --- Wetten simulieren ---
    kapital = bankroll
    verlauf, wetten = [], []
    for _, r in res.iterrows():
        for k in range(3):
            p, q = r["p_modell"][k], r["quoten"][k]
            edge = p * q - 1
            if edge < min_edge:
                continue
            if einsatz_modus == "kelly":
                stake = kapital * kelly(p, q)
            else:
                stake = bankroll * 0.01
            if stake <= 0:
                continue
            gewinn = stake * (q - 1) if r["ergebnis"] == k else -stake
            kapital += gewinn
            wetten.append({"stake": stake, "gewinn": gewinn, "edge": edge,
                           "quote": q, "treffer": int(r["ergebnis"] == k)})
            verlauf.append(kapital)

    w = pd.DataFrame(wetten)
    return {
        "spiele": len(res),
        "zeitraum": (res["Date"].min().date(), res["Date"].max().date()),
        "logloss_modell": ll_modell,
        "logloss_markt": ll_markt,
        "wetten": len(w),
        "umsatz": float(w["stake"].sum()) if len(w) else 0.0,
        "profit": float(w["gewinn"].sum()) if len(w) else 0.0,
        "roi": float(w["gewinn"].sum() / w["stake"].sum() * 100) if len(w) else 0.0,
        "trefferquote": float(w["treffer"].mean() * 100) if len(w) else 0.0,
        "endkapital": kapital,
        "verlauf": verlauf,
    }


# --------------------------------------------------------------------------
# 6. Export: alle Ligen in eine HTML-Datei
# --------------------------------------------------------------------------

def export_html(saisons, ligen=None, xi=0.0018, ziel="wettrechner_alle.html"):
    """Rechnet jede Liga durch und schreibt ein fertiges Browser-Tool."""
    ligen = ligen or list(LIGEN.keys())
    daten = {}

    for code in ligen:
        name = LIGEN.get(code, code)
        print(f"\n--- {code}  {name} ---")
        try:
            df = lade_daten(code, saisons)
        except SystemExit:
            print("  uebersprungen (keine Daten)")
            continue
        except Exception as e:
            print(f"  uebersprungen ({e})")
            continue

        if len(df) < 200:
            print(f"  uebersprungen (nur {len(df)} Spiele)")
            continue

        print(f"  {len(df)} Spiele, schaetze Parameter ...")
        m = DixonColes(xi=xi).fit(df)
        n = m.n
        att, deff = m.params[:n], m.params[n:2 * n]

        daten[code] = {
            "name": name,
            "stand": str(df["Date"].max().date()),
            "gamma": round(float(m.params[2 * n]), 4),
            "rho": round(float(m.params[2 * n + 1]), 4),
            "teams": {t: [round(float(att[i]), 4), round(float(deff[i]), 4)]
                      for t, i in m.teams.items()},
        }
        print(f"  fertig: {n} Teams, Heimvorteil {m.params[2*n]:+.3f}")

    if not daten:
        sys.exit("Keine Liga konnte verarbeitet werden.")

    html = HTML_TEMPLATE.replace("/*__DATEN__*/{}",
                                 json.dumps(daten, ensure_ascii=False))
    with open(ziel, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\n{'='*62}")
    print(f"{ziel} geschrieben — {len(daten)} Ligen, "
          f"{sum(len(d['teams']) for d in daten.values())} Teams")
    print("Datei doppelklicken, laeuft im Browser ohne Internet.")


# --------------------------------------------------------------------------
# 5. Ausgabe
# --------------------------------------------------------------------------

def zeige_prognose(modell, heim, ausw, quoten=None):
    pr = modell.prognose(heim, ausw)
    if pr is None:
        print(f"Team unbekannt: {heim} / {ausw}")
        print("Verfuegbar:", ", ".join(sorted(modell.teams)))
        return

    print(f"\n{heim} vs {ausw}")
    print(f"  Erwartete Tore: {pr['xg_heim']:.2f} : {pr['xg_ausw']:.2f}")
    print(f"  Heimsieg        {pr['heimsieg']*100:5.1f} %   (faire Quote {1/pr['heimsieg']:.2f})")
    print(f"  Unentschieden   {pr['unentschieden']*100:5.1f} %   (faire Quote {1/pr['unentschieden']:.2f})")
    print(f"  Auswaertssieg   {pr['auswaertssieg']*100:5.1f} %   (faire Quote {1/pr['auswaertssieg']:.2f})")
    print(f"  Ueber 2.5 Tore  {pr['ueber_2_5']*100:5.1f} %")
    print(f"  Beide treffen   {pr['btts']*100:5.1f} %")
    print("  Wahrscheinlichste Ergebnisse:",
          ", ".join(f"{i}:{j} ({p*100:.1f}%)" for (i, j), p in pr["top_ergebnisse"]))

    if quoten:
        print("\n  Value-Check:")
        namen = ["Heimsieg", "Remis", "Auswaerts"]
        ps = [pr["heimsieg"], pr["unentschieden"], pr["auswaertssieg"]]
        for name, p, q in zip(namen, ps, quoten):
            edge = p * q - 1
            mark = " <-- VALUE" if edge > 0.05 else ""
            print(f"    {name:10s} Quote {q:5.2f}  Edge {edge*100:+6.1f} %"
                  f"  Kelly {kelly(p, q)*100:4.1f} %{mark}")


def zeige_backtest(r):
    print("\n" + "=" * 62)
    print("BACKTEST")
    print("=" * 62)
    print(f"Zeitraum          {r['zeitraum'][0]} bis {r['zeitraum'][1]}")
    print(f"Spiele geprueft   {r['spiele']}")
    print()
    print(f"Log-Loss Modell   {r['logloss_modell']:.4f}")
    print(f"Log-Loss Markt    {r['logloss_markt']:.4f}   <-- der Massstab")
    diff = r["logloss_markt"] - r["logloss_modell"]
    if diff > 0:
        print(f"  Modell ist um {diff:.4f} besser kalibriert als der Markt.")
        print("  Das ist selten. Pruefe auf Datenlecks, bevor du dich freust.")
    else:
        print(f"  Markt ist um {-diff:.4f} besser. Das ist der Normalfall.")
    print()
    print(f"Value-Wetten      {r['wetten']}")
    print(f"Umsatz            {r['umsatz']:.2f}")
    print(f"Profit            {r['profit']:+.2f}")
    print(f"ROI               {r['roi']:+.2f} %")
    print(f"Trefferquote      {r['trefferquote']:.1f} %")
    print(f"Endkapital        {r['endkapital']:.2f}")
    print()
    n = r["wetten"]
    if n > 0:
        se = 100 * np.sqrt(1.0 / n) * 2
        print(f"Faustregel Rauschen: bei {n} Wetten liegt der Zufallsbereich")
        print(f"grob bei +/- {se:.1f} % ROI. Alles darunter ist kein Beweis.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--liga", default="D1")
    ap.add_argument("--saisons", nargs="+",
                    default=["2021", "2122", "2223", "2324", "2425", "2526"])
    ap.add_argument("--xi", type=float, default=0.0018)
    ap.add_argument("--backtest", action="store_true")
    ap.add_argument("--min-edge", type=float, default=0.05)
    ap.add_argument("--prognose", nargs=2, metavar=("HEIM", "AUSW"))
    ap.add_argument("--quoten", nargs=3, type=float, metavar=("H", "X", "A"))
    ap.add_argument("--teams", action="store_true")
    ap.add_argument("--export", action="store_true",
                    help="alle Ligen rechnen und HTML-Tool erzeugen")
    ap.add_argument("--nur", nargs="+", metavar="CODE",
                    help="Export auf diese Ligacodes beschraenken")
    args = ap.parse_args()

    if args.export:
        export_html(args.saisons, ligen=args.nur, xi=args.xi)
        return

    print(f"Liga {args.liga}, Saisons {' '.join(args.saisons)}")
    df = lade_daten(args.liga, args.saisons)
    print(f"{len(df)} Spiele geladen "
          f"({df['Date'].min().date()} bis {df['Date'].max().date()})")

    if args.backtest:
        zeige_backtest(backtest(df, xi=args.xi, min_edge=args.min_edge))
        return

    modell = DixonColes(xi=args.xi).fit(df)

    if args.teams:
        n = modell.n
        att, deff = modell.params[:n], modell.params[n:2 * n]
        tab = sorted(modell.teams.items(), key=lambda kv: -att[kv[1]])
        print(f"\n{'Team':24s} {'Angriff':>8s} {'Abwehr':>8s}")
        for t, i in tab:
            print(f"{t:24s} {att[i]:+8.3f} {deff[i]:+8.3f}")
        print(f"\nHeimvorteil {modell.params[2*n]:+.3f}, "
              f"rho {modell.params[2*n+1]:+.3f}")
        return

    if args.prognose:
        zeige_prognose(modell, args.prognose[0], args.prognose[1], args.quoten)
    else:
        print("\nNichts zu tun. Nutze --prognose, --backtest oder --teams.")


if __name__ == "__main__":
    main()
