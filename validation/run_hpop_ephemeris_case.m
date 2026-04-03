function run_hpop_ephemeris_case(case_dir, hpop_root)
%RUN_HPOP_EPHEMERIS_CASE Export DE440-based Sun/Moon ephemeris samples for bridge injection.

if nargin < 1 || strlength(string(case_dir)) == 0
    error('run_hpop_ephemeris_case:missingCaseDir', 'case_dir is required.');
end
if nargin < 2 || strlength(string(hpop_root)) == 0
    hpop_root = pwd;
end

case_dir = char(string(case_dir));
hpop_root = char(string(hpop_root));
case_input_path = fullfile(case_dir, 'hpop_ephemeris_input.m');
if ~isfile(case_input_path)
    error('run_hpop_ephemeris_case:missingInput', 'Missing case input file: %s', case_input_path);
end

old_dir = pwd;
cleanup_obj = onCleanup(@() cd(old_dir)); %#ok<NASGU>
cd(hpop_root);

global const eopdata PC

run(case_input_path);
if ~exist('case_params', 'var')
    error('run_hpop_ephemeris_case:missingCaseParams', 'Generated input did not define case_params.');
end

SAT_Const
constants
load(fullfile(hpop_root, 'DE440Coeff.mat'));
PC = DE440Coeff;

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

Mjd0_UTC = Mjday(case_params.epoch_year, case_params.epoch_month, case_params.epoch_day, ...
                 case_params.epoch_hour, case_params.epoch_minute, case_params.epoch_second);
Step = case_params.step_s;
N_Step = round(case_params.duration_s / case_params.step_s);

out_path = fullfile(case_dir, 'BodyEphemeris.txt');
fid = fopen(out_path, 'w');
for i = 0:N_Step
    t = i * Step;
    MJD_UTC = Mjd0_UTC + t / 86400;
    [~,~,UT1_UTC,~,~,~,~,~,TAI_UTC] = IERS(eopdata, MJD_UTC, 'l');
    [~,~,~,TT_UTC,~] = timediff(UT1_UTC, TAI_UTC);
    MJD_TT = MJD_UTC + TT_UTC / 86400;
    MJD_TDB = Mjday_TDB(MJD_TT);
    [~,~,~,~,~,~,~,~,~,r_Moon,r_Sun,~] = JPL_Eph_DE440(MJD_TDB + 2400000.5);
    fprintf(fid, '%.9f %.12f %.12f %.12f %.12f %.12f %.12f\n', ...
        t, r_Moon(1)/1e3, r_Moon(2)/1e3, r_Moon(3)/1e3, r_Sun(1)/1e3, r_Sun(2)/1e3, r_Sun(3)/1e3);
end
fclose(fid);
end
