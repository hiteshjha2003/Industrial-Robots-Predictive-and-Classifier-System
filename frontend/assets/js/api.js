/* frontend/assets/js/api.js */
const BASE_URL = 'http://localhost:8000';

const API = {
    async health() {
        const res = await fetch(`${BASE_URL}/health`);
        return res.json();
    },

    async generateData(mode = 'sample') {
        const res = await fetch(`${BASE_URL}/generate-data?mode=${mode}`, { method: 'POST' });
        return res.json();
    },

    async cleanData(mode = 'sample') {
        const res = await fetch(`${BASE_URL}/clean-data?mode=${mode}`, { method: 'POST' });
        return res.json();
    },

    async buildFeatures() {
        const res = await fetch(`${BASE_URL}/build-features`, { method: 'POST' });
        return res.json();
    },

    async train(mode = 'sample') {
        const res = await fetch(`${BASE_URL}/train?mode=${mode}`, { method: 'POST' });
        return res.json();
    },

    async predict(data) {
        const res = await fetch(`${BASE_URL}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        return res.json();
    },

    async getDataSummary(mode = 'sample') {
        const res = await fetch(`${BASE_URL}/data-summary?mode=${mode}`);
        return res.json();
    },

    async getTelemetrySample(robotId = 0, jointId = 0, mode = 'sample') {
        const res = await fetch(`${BASE_URL}/telemetry-sample?robot_id=${robotId}&joint_id=${jointId}&mode=${mode}`);
        return res.json();
    },

    async getTaskStatus(taskId) {
        const res = await fetch(`${BASE_URL}/task-status/${taskId}`);
        return res.json();
    }
};
