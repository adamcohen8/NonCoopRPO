function run_hpop_accel_case(case_dir, hpop_root)
%RUN_HPOP_ACCEL_CASE Compute one HPOP acceleration sample for a generated case.

if nargin < 1 || strlength(string(case_dir)) == 0
    error('run_hpop_accel_case:missingCaseDir', 'case_dir is required.');
end
if nargin < 2 || strlength(string(hpop_root)) == 0
    hpop_root = pwd;
end

case_dir = char(string(case_dir));
hpop_root = char(string(hpop_root));
case_input_path = fullfile(case_dir, 'hpop_case_input.m');
if ~isfile(case_input_path)
    error('run_hpop_accel_case:missingInput', 'Missing case input file: %s', case_input_path);
end

old_dir = pwd;
cleanup_obj = onCleanup(@() cd(old_dir)); %#ok<NASGU>
cd(hpop_root);

run(case_input_path);
if ~exist('case_params', 'var')
    error('run_hpop_accel_case:missingCaseParams', 'Generated input did not define case_params.');
end

global const Cnm Snm AuxParam eopdata swdata SOLdata DTCdata APdata PC

SAT_Const
constants
load(fullfile(hpop_root, 'DE440Coeff.mat'));
PC = DE440Coeff;

Cnm = zeros(361,361);
Snm = zeros(361,361);
fid = fopen(fullfile(hpop_root, 'GGM03C.txt'), 'r');
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
                  'planets',0,'SolidEarthTides',0,'OceanTides',0,'Relativity',0);

Mjd0_UTC = Mjday(case_params.epoch_year, case_params.epoch_month, case_params.epoch_day, ...
                 case_params.epoch_hour, case_params.epoch_minute, case_params.epoch_second);
Y0 = case_params.initial_state_eci_m_m_s(:);

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

MJD_UTC = AuxParam.Mjd_UTC;
[x_pole,y_pole,UT1_UTC,~,~,~,~,~,TAI_UTC] = IERS(eopdata,MJD_UTC,'l');
[~,~,~,TT_UTC,~] = timediff(UT1_UTC,TAI_UTC);
MJD_UT1 = MJD_UTC + UT1_UTC/86400;
MJD_TT  = MJD_UTC + TT_UTC/86400;
NPB = iauPnm06a(const.DJM0, MJD_TT);
gast = iauGst06(const.DJM0, MJD_UT1, const.DJM0, MJD_TT, NPB);
Theta  = iauRz(gast, eye(3));
Pi = iauPom00(x_pole, y_pole, iauSp00(const.DJM0, MJD_TT));
E = Pi*Theta*NPB;

r_bf = E * Y0(1:3);
a_harmonic_eci = AccelHarmonic(Y0(1:3), E, AuxParam.n, AuxParam.m);
a_harmonic_bf = E * a_harmonic_eci;
dY = Accel(0, Y0);
a = dY(4:6);

result_path = fullfile(case_dir, 'hpop_accel_result.txt');
fid = fopen(result_path, 'w');
fprintf(fid, 'ax_m_s2=%.15e\n', a(1));
fprintf(fid, 'ay_m_s2=%.15e\n', a(2));
fprintf(fid, 'az_m_s2=%.15e\n', a(3));
fprintf(fid, 'ax_harmonic_m_s2=%.15e\n', a_harmonic_eci(1));
fprintf(fid, 'ay_harmonic_m_s2=%.15e\n', a_harmonic_eci(2));
fprintf(fid, 'az_harmonic_m_s2=%.15e\n', a_harmonic_eci(3));
fprintf(fid, 'ax_bf_harmonic_m_s2=%.15e\n', a_harmonic_bf(1));
fprintf(fid, 'ay_bf_harmonic_m_s2=%.15e\n', a_harmonic_bf(2));
fprintf(fid, 'az_bf_harmonic_m_s2=%.15e\n', a_harmonic_bf(3));
fprintf(fid, 'rx_bf_m=%.15e\n', r_bf(1));
fprintf(fid, 'ry_bf_m=%.15e\n', r_bf(2));
fprintf(fid, 'rz_bf_m=%.15e\n', r_bf(3));
for row = 1:3
    for col = 1:3
        fprintf(fid, 'E_%d_%d=%.15e\n', row, col, E(row,col));
    end
end
fclose(fid);
end
