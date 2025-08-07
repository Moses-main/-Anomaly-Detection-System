// import { useState } from "react";
import "./App.css";

// function App() {
//   const [count, setCount] = useState(0);

//   return (
//     <>
//       <h2 className="text-red-700">hello</h2>
//     </>
//   );
// }

// export default App;

import React, { useState, useEffect, useRef } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  ScatterChart,
  Scatter,
} from "recharts";
import {
  Shield,
  AlertTriangle,
  Activity,
  TrendingUp,
  Users,
  Globe,
  Clock,
  Database,
  Play,
  Pause,
  Settings,
  Download,
  Filter,
} from "lucide-react";

const App =
  // NetworkAnomalyDetectionApp
  () => {
    const [isMonitoring, setIsMonitoring] = useState(false);
    const [alerts, setAlerts] = useState([]);
    const [networkData, setNetworkData] = useState([]);
    const [anomalyData, setAnomalyData] = useState([]);
    const [stats, setStats] = useState({
      totalConnections: 0,
      anomaliesDetected: 0,
      falsePositiveRate: 0,
      detectionAccuracy: 0,
    });
    const [selectedModel, setSelectedModel] = useState("ensemble");
    const [threshold, setThreshold] = useState(0.7);
    const intervalRef = useRef(null);

    // Simulated network traffic data generator
    const generateNetworkData = () => {
      const timestamp = new Date().toLocaleTimeString();
      const baseTraffic = Math.random() * 100 + 50;
      const anomalyProbability = Math.random();
      const isAnomaly = anomalyProbability > 0.85; // 15% chance of anomaly

      const newDataPoint = {
        time: timestamp,
        traffic: isAnomaly ? baseTraffic * (1.5 + Math.random()) : baseTraffic,
        anomalyScore: isAnomaly
          ? 0.8 + Math.random() * 0.2
          : Math.random() * 0.3,
        isAnomaly: isAnomaly,
        connections: Math.floor(Math.random() * 50) + 10,
        packetLoss: Math.random() * 5,
        latency: Math.random() * 100 + 20,
      };

      // Update network data (keep last 20 points)
      setNetworkData((prev) => {
        const updated = [...prev, newDataPoint];
        return updated.slice(-20);
      });

      // Update anomaly data if anomaly detected
      if (isAnomaly) {
        const newAlert = {
          id: Date.now(),
          timestamp: new Date().toLocaleString(),
          type: "High Risk",
          severity: Math.random() > 0.5 ? "Critical" : "High",
          source: `192.168.${Math.floor(Math.random() * 255)}.${Math.floor(
            Math.random() * 255
          )}`,
          destination: `10.0.${Math.floor(Math.random() * 255)}.${Math.floor(
            Math.random() * 255
          )}`,
          anomalyScore: newDataPoint.anomalyScore,
          description: "Suspicious traffic pattern detected",
        };

        setAlerts((prev) => [newAlert, ...prev.slice(0, 9)]);

        setAnomalyData((prev) => {
          const updated = [
            ...prev,
            {
              time: timestamp,
              score: newDataPoint.anomalyScore,
              type: newAlert.severity,
            },
          ];
          return updated.slice(-15);
        });
      }

      // Update statistics
      setStats((prev) => ({
        totalConnections: prev.totalConnections + newDataPoint.connections,
        anomaliesDetected: isAnomaly
          ? prev.anomaliesDetected + 1
          : prev.anomaliesDetected,
        falsePositiveRate: Math.max(
          0,
          prev.falsePositiveRate + (Math.random() - 0.5) * 0.1
        ),
        detectionAccuracy: 92 + Math.random() * 6,
      }));
    };

    // Start/Stop monitoring
    const toggleMonitoring = () => {
      if (isMonitoring) {
        clearInterval(intervalRef.current);
        setIsMonitoring(false);
      } else {
        setIsMonitoring(true);
        intervalRef.current = setInterval(generateNetworkData, 2000);
      }
    };

    // Initialize with some sample data
    useEffect(() => {
      // Generate initial data
      for (let i = 0; i < 10; i++) {
        setTimeout(() => generateNetworkData(), i * 200);
      }

      return () => {
        if (intervalRef.current) {
          clearInterval(intervalRef.current);
        }
      };
    }, []);

    // Sample protocol distribution data
    const protocolData = [
      { name: "TCP", value: 65, color: "#8884d8" },
      { name: "UDP", value: 25, color: "#82ca9d" },
      { name: "ICMP", value: 7, color: "#ffc658" },
      { name: "Other", value: 3, color: "#ff7300" },
    ];

    // Sample threat types data
    const threatData = [
      { name: "DDoS", count: 15 },
      { name: "Port Scan", count: 8 },
      { name: "Brute Force", count: 5 },
      { name: "Data Exfiltration", count: 3 },
      { name: "Malware", count: 2 },
    ];

    const StatCard = ({ title, value, icon: Icon, color, change }) => (
      <div
        className="bg-white rounded-lg shadow-md p-6 border-l-4"
        style={{ borderLeftColor: color }}
      >
        <div className="flex items-center justify-between">
          <div>
            <p className="text-sm font-medium text-gray-600">{title}</p>
            <p className="text-2xl font-bold text-gray-900">{value}</p>
            {change && (
              <p
                className={`text-sm ${
                  change > 0 ? "text-green-600" : "text-red-600"
                }`}
              >
                {change > 0 ? "+" : ""}
                {change.toFixed(1)}%
              </p>
            )}
          </div>
          <Icon className="h-8 w-8" style={{ color }} />
        </div>
      </div>
    );

    const AlertItem = ({ alert }) => (
      <div
        className={`p-4 border-l-4 mb-3 rounded-r-lg ${
          alert.severity === "Critical"
            ? "bg-red-50 border-red-500"
            : "bg-orange-50 border-orange-500"
        }`}
      >
        <div className="flex justify-between items-start">
          <div>
            <h4 className="font-semibold text-gray-900">{alert.type}</h4>
            <p className="text-sm text-gray-600">{alert.description}</p>
            <div className="mt-2 text-xs text-gray-500">
              <span>Source: {alert.source}</span> •{" "}
              <span>Dest: {alert.destination}</span>
            </div>
          </div>
          <div className="text-right">
            <span
              className={`px-2 py-1 rounded text-xs font-medium ${
                alert.severity === "Critical"
                  ? "bg-red-100 text-red-800"
                  : "bg-orange-100 text-orange-800"
              }`}
            >
              {alert.severity}
            </span>
            <p className="text-xs text-gray-500 mt-1">{alert.timestamp}</p>
            <p className="text-xs font-mono">
              Score: {alert.anomalyScore.toFixed(3)}
            </p>
          </div>
        </div>
      </div>
    );

    return (
      <div className="min-h-screen bg-gray-100">
        {/* Header */}
        <div className="bg-white shadow-sm border-b">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex justify-between items-center py-6">
              <div className="flex items-center">
                <Shield className="h-8 w-8 text-blue-600 mr-3" />
                <h1 className="text-2xl font-bold text-gray-900">
                  Network Anomaly Detection System
                </h1>
              </div>
              <div className="flex items-center space-x-4">
                <select
                  value={selectedModel}
                  onChange={(e) => setSelectedModel(e.target.value)}
                  className="px-3 py-2 border border-gray-300 rounded-md text-sm"
                >
                  <option value="isolation_forest">Isolation Forest</option>
                  <option value="autoencoder">Autoencoder</option>
                  <option value="ensemble">Ensemble Model</option>
                </select>
                <button
                  onClick={toggleMonitoring}
                  className={`flex items-center px-4 py-2 rounded-md text-white font-medium ${
                    isMonitoring
                      ? "bg-red-600 hover:bg-red-700"
                      : "bg-green-600 hover:bg-green-700"
                  }`}
                >
                  {isMonitoring ? (
                    <Pause className="h-4 w-4 mr-2 text-red-700" />
                  ) : (
                    <Play className="h-4 w-4 mr-2 text-green-600" />
                  )}
                  {isMonitoring ? "Stop Monitoring" : "Start Monitoring"}
                </button>
              </div>
            </div>
          </div>
        </div>

        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          {/* Status Indicators */}
          <div className="mb-8">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-semibold text-gray-900">
                System Status
              </h2>
              <div className="flex items-center space-x-2">
                <div
                  className={`w-3 h-3 rounded-full ${
                    isMonitoring ? "bg-green-500" : "bg-red-500"
                  }`}
                ></div>
                <span className="text-sm text-gray-600">
                  {isMonitoring ? "Monitoring Active" : "Monitoring Inactive"}
                </span>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              <StatCard
                title="Total Connections"
                value={stats.totalConnections.toLocaleString()}
                icon={Globe}
                color="#3b82f6"
                change={2.4}
              />
              <StatCard
                title="Anomalies Detected"
                value={stats.anomaliesDetected}
                icon={AlertTriangle}
                color="#ef4444"
                change={-1.2}
              />
              <StatCard
                title="Detection Accuracy"
                value={`${stats.detectionAccuracy.toFixed(1)}%`}
                icon={TrendingUp}
                color="#10b981"
                change={0.3}
              />
              <StatCard
                title="False Positive Rate"
                value={`${Math.max(0, stats.falsePositiveRate).toFixed(2)}%`}
                icon={Activity}
                color="#f59e0b"
                change={-0.1}
              />
            </div>
          </div>

          {/* Main Dashboard Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Network Traffic Chart */}
            <div className="lg:col-span-2">
              <div className="bg-white p-6 rounded-lg shadow-md">
                <div className="flex justify-between items-center mb-4">
                  <h3 className="text-lg font-semibold text-gray-900">
                    Network Traffic Monitor
                  </h3>
                  <div className="flex items-center space-x-2">
                    <label className="text-sm text-gray-600">Threshold:</label>
                    <input
                      type="range"
                      min="0.1"
                      max="1"
                      step="0.1"
                      value={threshold}
                      onChange={(e) => setThreshold(parseFloat(e.target.value))}
                      className="w-20"
                    />
                    <span className="text-sm text-gray-600">{threshold}</span>
                  </div>
                </div>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={networkData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="time" tick={{ fontSize: 12 }} />
                    <YAxis />
                    <Tooltip />
                    <Line
                      type="monotone"
                      dataKey="traffic"
                      stroke="#3b82f6"
                      strokeWidth={2}
                      dot={{ fill: "#3b82f6", r: 3 }}
                    />
                    <Line
                      type="monotone"
                      dataKey="anomalyScore"
                      stroke="#ef4444"
                      strokeWidth={2}
                      strokeDasharray="5 5"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Recent Alerts */}
            <div className="bg-white p-6 rounded-lg shadow-md">
              <div className="flex justify-between items-center mb-4">
                <h3 className="text-lg font-semibold text-gray-900">
                  Recent Alerts
                </h3>
                <Filter className="h-5 w-5 text-gray-400" />
              </div>
              <div className="max-h-96 overflow-y-auto">
                {alerts.length === 0 ? (
                  <p className="text-gray-500 text-center py-8">
                    No alerts detected
                  </p>
                ) : (
                  alerts.map((alert) => (
                    <AlertItem key={alert.id} alert={alert} />
                  ))
                )}
              </div>
            </div>
          </div>

          {/* Secondary Charts Row */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mt-6">
            {/* Protocol Distribution */}
            <div className="bg-white p-6 rounded-lg shadow-md">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">
                Protocol Distribution
              </h3>
              <ResponsiveContainer width="100%" height={200}>
                <PieChart>
                  <Pie
                    data={protocolData}
                    cx="50%"
                    cy="50%"
                    innerRadius={40}
                    outerRadius={80}
                    dataKey="value"
                    label={({ name, value }) => `${name}: ${value}%`}
                  >
                    {protocolData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>

            {/* Threat Types */}
            <div className="bg-white p-6 rounded-lg shadow-md">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">
                Threat Types
              </h3>
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={threatData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" tick={{ fontSize: 10 }} />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="count" fill="#ef4444" />
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Anomaly Scores Scatter */}
            <div className="bg-white p-6 rounded-lg shadow-md">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">
                Anomaly Scores
              </h3>
              <ResponsiveContainer width="100%" height={200}>
                <ScatterChart data={anomalyData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="time" tick={{ fontSize: 10 }} />
                  <YAxis domain={[0, 1]} />
                  <Tooltip />
                  <Scatter
                    dataKey="score"
                    fill={(entry) =>
                      entry.type === "Critical" ? "#ef4444" : "#f59e0b"
                    }
                  />
                </ScatterChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Model Configuration Panel */}
          <div className="mt-6 bg-white p-6 rounded-lg shadow-md">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-lg font-semibold text-gray-900">
                Model Configuration
              </h3>
              <div className="flex space-x-2">
                <button className="flex items-center px-3 py-2 bg-blue-600 text-white rounded-md text-sm hover:bg-blue-700">
                  <Download className="h-4 w-4 mr-2" />
                  Export Data
                </button>
                <button className="flex items-center px-3 py-2 bg-gray-600 text-white rounded-md text-sm hover:bg-gray-700">
                  <Settings className="h-4 w-4 mr-2" />
                  Settings
                </button>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Detection Model
                </label>
                <select
                  value={selectedModel}
                  onChange={(e) => setSelectedModel(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md"
                >
                  <option value="isolation_forest">Isolation Forest</option>
                  <option value="autoencoder">
                    Autoencoder Neural Network
                  </option>
                  <option value="ensemble">Ensemble (IF + AE)</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Sensitivity Threshold: {threshold}
                </label>
                <input
                  type="range"
                  min="0.1"
                  max="1"
                  step="0.1"
                  value={threshold}
                  onChange={(e) => setThreshold(parseFloat(e.target.value))}
                  className="w-full"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Model Status
                </label>
                <div className="flex items-center">
                  <div className="w-3 h-3 bg-green-500 rounded-full mr-2"></div>
                  <span className="text-sm text-gray-600">
                    Model Trained & Active
                  </span>
                </div>
              </div>
            </div>

            <div className="mt-4 pt-4 border-t border-gray-200">
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4 text-sm">
                <div>
                  <span className="font-medium text-gray-700">
                    Last Training:
                  </span>
                  <p className="text-gray-600">2 hours ago</p>
                </div>
                <div>
                  <span className="font-medium text-gray-700">
                    Training Data:
                  </span>
                  <p className="text-gray-600">1.2M samples</p>
                </div>
                <div>
                  <span className="font-medium text-gray-700">
                    Model Version:
                  </span>
                  <p className="text-gray-600">v2.1.0</p>
                </div>
                <div>
                  <span className="font-medium text-gray-700">
                    Next Retrain:
                  </span>
                  <p className="text-gray-600">In 22 hours</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  };

export default App;
// NetworkAnomalyDetectionApp;
