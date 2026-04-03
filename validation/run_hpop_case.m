function run_hpop_case(case_dir, hpop_root)
%RUN_HPOP_CASE Run a generated HPOP validation case without editing test_HPOP.m.

if nargin < 1 || strlength(string(case_dir)) == 0
    error('run_hpop_case:missingCaseDir', 'case_dir is required.');
end
if nargin < 2 || strlength(string(hpop_root)) == 0
    hpop_root = pwd;
end

case_dir = char(string(case_dir));
hpop_root = char(string(hpop_root));
case_input_path = fullfile(case_dir, 'hpop_case_input.m');
if ~isfile(case_input_path)
    error('run_hpop_case:missingInput', 'Missing case input file: %s', case_input_path);
end

old_dir = pwd;
cleanup_obj = onCleanup(@() cd(old_dir)); %#ok<NASGU>
cd(hpop_root);

global const Cnm Snm AuxParam eopdata swdata SOLdata DTCdata APdata PC RKF78_TRACE_PATH RKF78_STAGE_TRACE_PATH RKF78_STAGE_TRACE_WRITTEN

run(case_input_path);
if ~exist('case_params', 'var')
    error('run_hpop_case:missingCaseParams', 'Generated input did not define case_params.');
end
RKF78_TRACE_PATH = '';
RKF78_STAGE_TRACE_PATH = '';
RKF78_STAGE_TRACE_WRITTEN = 0;
if isfield(case_params, 'rkf78_trace_path')
    RKF78_TRACE_PATH = char(string(case_params.rkf78_trace_path));
    if ~isempty(RKF78_TRACE_PATH)
        fid_trace = fopen(RKF78_TRACE_PATH, 'w');
        fclose(fid_trace);
    end
end
if isfield(case_params, 'rkf78_stage_trace_path')
    RKF78_STAGE_TRACE_PATH = char(string(case_params.rkf78_stage_trace_path));
    if ~isempty(RKF78_STAGE_TRACE_PATH)
        fid_trace = fopen(RKF78_STAGE_TRACE_PATH, 'w');
        fclose(fid_trace);
    end
end

SAT_Const
constants
load(fullfile(fileparts(mfilename('fullpath')), 'data', 'DE440Coeff.mat'));
PC = DE440Coeff;

Cnm = zeros(361,361);
Snm = zeros(361,361);
fid = fopen(fullfile(fileparts(mfilename('fullpath')), 'data', 'GGM03C.txt'), 'r');
for n = 0:360
    for m = 0:n
        temp = fscanf(fid, '%d %d %f %f %f %f', [6 1]);
        Cnm(n+1,m+1) = temp(3);
        Snm(n+1,m+1) = temp(4);
    end
end
fclose(fid);

fid = fopen(fullfile(hpop_root, 'EOP-All.txt'), 'r');
while ~feof(fid)
    tline = fgetl(fid);
    k = strfind(tline, 'NUM_OBSERVED_POINTS');
    if (k ~= 0)
        numrecsobs = str2num(tline(21:end)); %#ok<ST2NM>
        tline = fgetl(fid); %#ok<NASGU>
        for i = 1:numrecsobs
            eopdata(:,i) = fscanf(fid, '%i %d %d %i %f %f %f %f %f %f %f %f %i', [13 1]); %#ok<AGROW>
        end
        for i = 1:4
            tline = fgetl(fid); %#ok<NASGU>
        end
        numrecspred = str2num(tline(22:end)); %#ok<ST2NM>
        tline = fgetl(fid); %#ok<NASGU>
        for i = numrecsobs+1:numrecsobs+numrecspred
            eopdata(:,i) = fscanf(fid, '%i %d %d %i %f %f %f %f %f %f %f %f %i', [13 1]); %#ok<AGROW>
        end
        break
    end
end
fclose(fid);

fid = fopen(fullfile(hpop_root, 'SW-All.txt'), 'r');
while ~feof(fid)
    tline = fgetl(fid);
    k = strfind(tline, 'NUM_OBSERVED_POINTS');
    if (k ~= 0)
        numrecsobs = str2num(tline(21:end)); %#ok<ST2NM>
        tline = fgetl(fid); %#ok<NASGU>
        for i = 1:numrecsobs
            swdata(:,i) = fscanf(fid, '%i %d %d %i %i %i %i %i %i %i %i %i %i %i %i %i %i %i %i %i %i %i %i %f %i %i %f %i %f %f %f %f %f', [33 1]); %#ok<AGROW>
        end
        swdata = [swdata(1:27,:); swdata(29:33,:)];
        for i = 1:4
            tline = fgetl(fid); %#ok<NASGU>
        end
        numrecspred = str2num(tline(28:end)); %#ok<ST2NM>
        tline = fgetl(fid); %#ok<NASGU>
        for i = numrecsobs+1:numrecsobs+numrecspred
            swdata(:,i) = fscanf(fid, '%i %d %d %i %i %i %i %i %i %i %i %i %i %i %i %i %i %i %i %i %i %i %i %f %i %i %f %f %f %f %f %f', [32 1]); %#ok<AGROW>
        end
        break
    end
end
fclose(fid);

fid = fopen(fullfile(hpop_root, 'SOLFSMY.txt'), 'r');
SOLdata = fscanf(fid, '%d %d %f %f %f %f %f %f %f %f %f', [11 inf]);
fclose(fid);

fid = fopen(fullfile(hpop_root, 'SOLRESAP.txt'), 'r');
APdata = fscanf(fid, '%d %d %d %d %d %d %d %d %d %d %d %d', [12 inf]);
fclose(fid);

fid = fopen(fullfile(hpop_root, 'DTCFILE.txt'), 'r');
DTCdata = fscanf(fid, '%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d', [26 inf]);
fclose(fid);

AuxParam = struct('Mjd_UTC',0,'area_solar',0,'area_drag',0,'mass',0,'Cr',0, ...
                  'Cd',0,'n',0,'m',0,'sun',0,'moon',0,'sRad',0,'drag',0, ...
                  'planets',0,'SolidEarthTides',0,'OceanTides',0,'Relativity',0, ...
                  'atmosphere_model','nrlmsise00');

Mjd0_UTC = Mjday(case_params.epoch_year, case_params.epoch_month, case_params.epoch_day, ...
                 case_params.epoch_hour, case_params.epoch_minute, case_params.epoch_second);
Y0 = case_params.initial_state_eci_m_m_s(:)';

AuxParam.Mjd_UTC = Mjd0_UTC;
AuxParam.area_solar = case_params.area_solar_m2;
AuxParam.area_drag = case_params.area_drag_m2;
AuxParam.mass = case_params.mass_kg;
AuxParam.Cr = case_params.cr;
AuxParam.Cd = case_params.cd;
AuxParam.n = case_params.gravity_degree;
AuxParam.m = case_params.gravity_order;
AuxParam.sun = case_params.enable_sun;
AuxParam.moon = case_params.enable_moon;
AuxParam.sRad = case_params.enable_solar_radiation;
AuxParam.drag = case_params.enable_drag;
AuxParam.planets = case_params.enable_planets;
AuxParam.SolidEarthTides = case_params.enable_solid_earth_tides;
AuxParam.OceanTides = case_params.enable_ocean_tides;
AuxParam.Relativity = case_params.enable_relativity;
if isfield(case_params, 'atmosphere_model')
    AuxParam.atmosphere_model = char(lower(string(case_params.atmosphere_model)));
end

Step = case_params.step_s;
N_Step = round(case_params.duration_s / case_params.step_s);

Eph_eci = Ephemeris(Y0, N_Step, Step);

states_path = fullfile(case_dir, 'SatelliteStates.txt');
fid = fopen(states_path, 'w');
for i = 1:N_Step+1
    [year,month,day,hour,minute,sec] = invjday(Mjd0_UTC + Eph_eci(i,1) / 86400);
    fprintf(fid, '  %4d/%2.2d/%2.2d  %2d:%2d:%12.9f', year, month, day, hour, minute, sec);
    fprintf(fid, '  %20.9f%20.9f%20.9f%18.9f%18.9f%18.9f\n', ...
            Eph_eci(i,2), Eph_eci(i,3), Eph_eci(i,4), Eph_eci(i,5), Eph_eci(i,6), Eph_eci(i,7));
end
fclose(fid);

result_path = fullfile(case_dir, 'hpop_case_result.txt');
fid = fopen(result_path, 'w');
fprintf(fid, 'states_path=%s\n', states_path);
fprintf(fid, 'samples=%d\n', N_Step + 1);
fprintf(fid, 'step_s=%.6f\n', Step);
fprintf(fid, 'duration_s=%.6f\n', N_Step * Step);
fclose(fid);
end
